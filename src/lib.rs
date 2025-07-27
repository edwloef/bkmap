use std::{iter::FusedIterator, marker::PhantomData, num::NonZero};

pub trait Metric<T: ?Sized> {
    fn distance(&mut self, a: &T, b: &T) -> usize;
}

#[derive(Debug, Default)]
pub struct Levenshtein<I, const INSERTION: usize, const DELETION: usize, const SUBSTITUTION: usize>
{
    cache: Vec<usize>,
    _i: PhantomData<I>,
}

impl<
    I: PartialEq,
    T: AsRef<[I]> + ?Sized,
    const INSERTION: usize,
    const DELETION: usize,
    const SUBSTITUTION: usize,
> Metric<T> for Levenshtein<I, INSERTION, DELETION, SUBSTITUTION>
{
    fn distance(&mut self, a: &T, b: &T) -> usize {
        let a = a.as_ref();
        let b = b.as_ref();

        self.cache.clear();
        self.cache.extend(1..=b.len());

        let mut result = b.len();

        for (mut last, a) in a.iter().enumerate() {
            result = last + 1;

            for (b, cache) in b.iter().zip(self.cache.iter_mut()) {
                let tmp = last + if a == b { 0 } else { SUBSTITUTION };
                last = *cache;
                result = tmp.min(last + DELETION).min(result + INSERTION);
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
    pub fn get_or_default(&mut self, key: K) -> &mut V
    where
        V: Default,
        M: Metric<K>,
    {
        if self.root.is_none() {
            self.root = Some(BKNode {
                dist: NonZero::new(1).unwrap(),
                key,
                value: V::default(),
                children: Vec::new(),
            });

            return &mut self.root.as_mut().unwrap().value;
        }

        let mut node = self.root.as_mut().unwrap();

        loop {
            let Some(dist) = NonZero::new(self.metric.distance(&key, &node.key)) else {
                return &mut node.value;
            };

            let Some(child) = node.children.iter().position(|child| child.dist == dist) else {
                node.children.push(BKNode {
                    dist,
                    key,
                    value: V::default(),
                    children: Vec::new(),
                });

                return &mut node.children.last_mut().unwrap().value;
            };

            node = &mut node.children[child];
        }
    }

    pub fn fuzzy<'a: 'b, 'b>(
        &'a self,
        key: &'b K,
        tolerance: usize,
    ) -> impl Iterator<Item = (usize, &'a K, &'a V)>
    where
        M: Metric<K> + Default,
    {
        BKFuzzy {
            metric: M::default(),
            stack: self.root.as_ref().into_iter().collect(),
            key,
            tolerance,
        }
    }
}

struct BKFuzzy<'a, 'b, K, V, M> {
    metric: M,
    stack: Vec<&'a BKNode<K, V>>,
    key: &'b K,
    tolerance: usize,
}

impl<'a: 'b, 'b, K, V, M: Metric<K>> Iterator for BKFuzzy<'a, 'b, K, V, M> {
    type Item = (usize, &'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.stack.pop()?;
            let dist = self.metric.distance(self.key, &node.key);

            self.stack.extend(
                node.children
                    .iter()
                    .filter(|child| child.dist.get().abs_diff(dist) <= self.tolerance),
            );

            if dist <= self.tolerance {
                return Some((dist, &node.key, &node.value));
            }
        }
    }
}

impl<'a: 'b, 'b, K, V, M: Metric<K>> FusedIterator for BKFuzzy<'a, 'b, K, V, M> {}
