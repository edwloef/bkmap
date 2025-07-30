use std::{iter::FusedIterator, marker::PhantomData, num::NonZero};

pub trait Metric<A, B> {
    fn distance(&mut self, a: A, b: B) -> usize;
}

#[derive(Debug)]
pub struct Levenshtein<E, const I: usize = 1, const D: usize = 1, const S: usize = 1> {
    cache: Vec<usize>,
    _e: PhantomData<E>,
}

impl<E, const I: usize, const D: usize, const S: usize> Default for Levenshtein<E, I, D, S> {
    fn default() -> Self {
        Self {
            cache: Vec::new(),
            _e: PhantomData,
        }
    }
}

impl<A: AsRef<[E]>, B: AsRef<[E]>, E: PartialEq, const I: usize, const D: usize, const S: usize>
    Metric<A, B> for Levenshtein<E, I, D, S>
{
    fn distance(&mut self, a: A, b: B) -> usize {
        let a = a.as_ref();
        let b = b.as_ref();

        self.cache.clear();
        self.cache.extend((1..).map(|x| x * I).take(b.len()));

        let mut result = b.len() * I;

        for (a, mut last) in a.iter().zip((0..).map(|x| x * D)) {
            result = last + D;

            for (b, cache) in b.iter().zip(self.cache.iter_mut()) {
                let tmp = last + if a == b { 0 } else { S };
                last = *cache;
                result = tmp.min(last + D).min(result + I);
                *cache = result;
            }
        }

        result
    }
}

#[derive(Debug)]
pub struct BKMap<K, V, M> {
    root: Option<BKNode<K, V>>,
    metric: M,
}

impl<K, V, M: Default> Default for BKMap<K, V, M> {
    fn default() -> Self {
        Self {
            root: None,
            metric: M::default(),
        }
    }
}

#[derive(Debug)]
struct BKNode<K, V> {
    dist: NonZero<usize>,
    key: K,
    value: V,
    children: Vec<BKNode<K, V>>,
}

impl<K, V, M> BKMap<K, V, M> {
    pub fn insert<'a>(&'a mut self, key: K, value: V)
    where
        M: for<'b> Metric<&'b K, &'a K>,
    {
        if self.root.is_none() {
            self.root = Some(BKNode {
                dist: NonZero::new(1).unwrap(),
                key,
                value,
                children: Vec::new(),
            });
            return;
        }

        let mut node = self.root.as_mut().unwrap();

        loop {
            let Some(dist) = NonZero::new(self.metric.distance(&key, &node.key)) else {
                node.value = value;
                return;
            };

            let Some(child) = node.children.iter().position(|child| child.dist == dist) else {
                node.children.push(BKNode {
                    dist,
                    key,
                    value,
                    children: Vec::new(),
                });
                return;
            };

            node = &mut node.children[child];
        }
    }

    pub fn fuzzy_search_distance<'a, S: Copy>(
        &'a self,
        key: S,
        distance: usize,
    ) -> impl Iterator<Item = (usize, &'a K, &'a V)>
    where
        M: Metric<S, &'a K> + Default,
    {
        BKFuzzy {
            metric: M::default(),
            stack: self.root.as_ref().into_iter().collect(),
            key,
            distance,
        }
    }

    pub fn fuzzy_search_count<'a, S: Copy>(
        &'a self,
        key: S,
        count: usize,
    ) -> Vec<(usize, &'a K, &'a V)>
    where
        M: Metric<S, &'a K> + Default,
    {
        let Some(root) = self.root.as_ref() else {
            return Vec::new();
        };

        let mut metric = M::default();

        let mut ret = Vec::with_capacity(count);
        let mut stack = vec![(0, root)];

        while let Some((dist, node)) = stack.pop() {
            let distance = ret.get(count - 1).map_or(usize::MAX, |(x, _, _)| *x);

            if node.dist.get().abs_diff(dist) > distance {
                continue;
            }

            let dist = metric.distance(key, &node.key);

            stack.extend(
                node.children
                    .iter()
                    .filter(|child| child.dist.get().abs_diff(dist) <= distance)
                    .map(|x| (dist, x)),
            );

            if dist <= distance {
                let i = ret.partition_point(|(x, _, _)| *x <= dist);
                ret.insert(i, (dist, &node.key, &node.value));
                if ret.len() > count && i < count && ret[count - 1].0 != ret[count].0 {
                    ret.truncate(count);
                }
            }
        }

        ret
    }
}

struct BKFuzzy<'a, K, V, M, S> {
    metric: M,
    stack: Vec<&'a BKNode<K, V>>,
    key: S,
    distance: usize,
}

impl<'a, K, V, M: Metric<S, &'a K>, S: Copy> Iterator for BKFuzzy<'a, K, V, M, S> {
    type Item = (usize, &'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.stack.pop()?;
            let dist = self.metric.distance(self.key, &node.key);

            self.stack.extend(
                node.children
                    .iter()
                    .filter(|child| child.dist.get().abs_diff(dist) <= self.distance),
            );

            if dist <= self.distance {
                return Some((dist, &node.key, &node.value));
            }
        }
    }
}

impl<K, V, M, S> FusedIterator for BKFuzzy<'_, K, V, M, S> where Self: Iterator {}
