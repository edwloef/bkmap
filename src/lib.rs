#![no_std]
extern crate alloc;

use alloc::vec::Vec;
use core::{iter::FusedIterator, marker::PhantomData, num::NonZero};

pub trait Metric<A, B> {
    fn distance(&mut self, a: A, b: B) -> usize;
}

#[derive(Debug)]
pub struct Levenshtein<E> {
    cache: Vec<usize>,
    _e: PhantomData<E>,
}

impl<E> Default for Levenshtein<E> {
    fn default() -> Self {
        Self {
            cache: Vec::new(),
            _e: PhantomData,
        }
    }
}

impl<A: AsRef<[E]>, B: AsRef<[E]>, E: PartialEq> Metric<A, B> for Levenshtein<E> {
    fn distance(&mut self, a: A, b: B) -> usize {
        let a = a.as_ref();
        let b = b.as_ref();

        self.cache.clear();
        self.cache.extend(1..=b.len());

        let mut result = b.len();

        for (mut last, a) in a.iter().enumerate() {
            result = last + 1;
            for (b, cache) in b.iter().zip(&mut self.cache) {
                result = (last + usize::from(a != b)).min(*cache + 1).min(result + 1);
                (last, *cache) = (*cache, result);
            }
        }

        result
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug)]
pub struct BKMap<K, V, M> {
    root: Option<BKNode<K, V>>,
    #[cfg_attr(feature = "serde", serde(skip))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug)]
struct BKNode<K, V> {
    dist: NonZero<usize>,
    key: K,
    value: V,
    children: Vec<Self>,
}

impl<K, V> BKNode<K, V> {
    fn shrink_to_fit(&mut self) {
        self.children.shrink_to_fit();

        for child in &mut self.children {
            child.shrink_to_fit();
        }
    }

    fn children_around(&self, dist: usize, distance: usize) -> impl Iterator<Item = &Self> {
        self.children
            .iter()
            .skip_while(move |child| child.dist.get() < dist.saturating_sub(distance))
            .take_while(move |child| child.dist.get() <= dist.saturating_add(distance))
    }
}

impl<K, V, M> BKMap<K, V, M> {
    pub fn insert<'a>(&'a mut self, key: K, value: V)
    where
        M: for<'b> Metric<&'b K, &'a K>,
    {
        if self.root.is_none() {
            return self.root = Some(BKNode {
                dist: NonZero::new(1).unwrap(),
                key,
                value,
                children: Vec::new(),
            });
        }

        let mut node = self.root.as_mut().unwrap();

        loop {
            let Some(dist) = NonZero::new(self.metric.distance(&key, &node.key)) else {
                return node.value = value;
            };

            let child = node.children.iter().position(|child| child.dist >= dist);

            let Some(child) = child.filter(|child| node.children[*child].dist == dist) else {
                return node.children.insert(
                    child.unwrap_or(node.children.len()),
                    BKNode {
                        dist,
                        key,
                        value,
                        children: Vec::new(),
                    },
                );
            };

            node = &mut node.children[child];
        }
    }

    pub fn shrink_to_fit(&mut self) {
        if let Some(root) = &mut self.root {
            root.shrink_to_fit();
        }
    }

    pub fn fuzzy_search_distance<'a, S: Copy>(
        &'a self,
        key: S,
        distance: usize,
    ) -> BKFuzzy<'a, K, V, M, S>
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
        let mut stack = Vec::from([(0, root)]);

        while let Some((dist, node)) = stack.pop() {
            let distance = ret.get(count - 1).map_or(usize::MAX, |(x, _, _)| *x);

            if node.dist.get().abs_diff(dist) > distance {
                continue;
            }

            let dist = metric.distance(key, &node.key);

            stack.extend(
                node.children_around(dist, distance)
                    .map(|child| (dist, child)),
            );

            if dist <= distance {
                let i = ret
                    .iter()
                    .position(|(x, _, _)| *x > dist)
                    .unwrap_or(ret.len());
                ret.insert(i, (dist, &node.key, &node.value));
                if ret.len() > count && i < count && ret[count - 1].0 != ret[count].0 {
                    ret.truncate(count);
                }
            }
        }

        ret
    }
}

#[derive(Debug)]
pub struct BKFuzzy<'a, K, V, M, S> {
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

            self.stack.extend(node.children_around(dist, self.distance));

            if dist <= self.distance {
                return Some((dist, &node.key, &node.value));
            }
        }
    }
}

impl<K, V, M, S> FusedIterator for BKFuzzy<'_, K, V, M, S> where Self: Iterator {}
