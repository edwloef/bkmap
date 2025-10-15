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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
struct BKNode<K, V> {
    dist: NonZero<usize>,
    key: K,
    value: V,
    children: Vec<Self>,
}

impl<K, V> BKNode<K, V> {
    fn len(&self) -> usize {
        self.children.iter().map(Self::len).sum::<usize>() + 1
    }

    fn capacity(&self) -> usize {
        self.children.iter().map(Self::capacity).sum::<usize>() + self.children.capacity()
    }

    fn shrink_to_fit(&mut self) {
        self.children.shrink_to_fit();

        for child in &mut self.children {
            child.shrink_to_fit();
        }
    }

    fn children_around(&self, dist: usize, radius: usize) -> impl Iterator<Item = &Self> {
        self.children
            .iter()
            .skip_while(move |child| child.dist.get() < dist.saturating_sub(radius))
            .take_while(move |child| child.dist.get() <= dist.saturating_add(radius))
    }
}

impl<K, V, M> BKMap<K, V, M> {
    pub fn insert<'a>(&'a mut self, key: K, value: V)
    where
        M: for<'b> Metric<&'b K, &'a K>,
    {
        self.insert_or_modify(key, value, |old, new| *old = new);
    }

    pub fn insert_or_modify<'a>(&'a mut self, key: K, value: V, modify: impl FnOnce(&mut V, V))
    where
        M: for<'b> Metric<&'b K, &'a K>,
    {
        self.insert_and_modify(key, value, |old, new| {
            if let Some(new) = new {
                modify(old, new);
            }
        });
    }

    pub fn insert_and_modify<'a>(
        &'a mut self,
        key: K,
        mut value: V,
        modify: impl FnOnce(&mut V, Option<V>),
    ) where
        M: for<'b> Metric<&'b K, &'a K>,
    {
        if self.root.is_none() {
            modify(&mut value, None);
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
                return modify(&mut node.value, Some(value));
            };

            let child = node.children.iter().position(|child| child.dist >= dist);

            let Some(child) = child.filter(|child| node.children[*child].dist == dist) else {
                modify(&mut value, None);
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

    #[must_use]
    pub fn len(&self) -> usize {
        self.root.as_ref().map_or(0, BKNode::len)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.root.as_ref().map_or(0, BKNode::capacity)
    }

    pub fn shrink_to_fit(&mut self) {
        if let Some(root) = &mut self.root {
            root.shrink_to_fit();
        }
    }

    pub fn fuzzy_search_distance<'a, S>(
        &'a self,
        key: S,
        distance: usize,
    ) -> BKFuzzy<'a, K, V, M, S>
    where
        M: for<'b> Metric<&'b S, &'a K> + Default,
    {
        BKFuzzy {
            metric: M::default(),
            stack: self.root.as_ref().into_iter().collect(),
            key,
            distance,
        }
    }
}

#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct BKFuzzy<'a, K, V, M, S> {
    metric: M,
    stack: Vec<&'a BKNode<K, V>>,
    key: S,
    distance: usize,
}

impl<'a, K, V, M: for<'b> Metric<&'b S, &'a K>, S> Iterator for BKFuzzy<'a, K, V, M, S> {
    type Item = (usize, &'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.stack.pop()?;
            let dist = self.metric.distance(&self.key, &node.key);

            self.stack.extend(node.children_around(dist, self.distance));

            if dist <= self.distance {
                return Some((dist, &node.key, &node.value));
            }
        }
    }
}

impl<K, V, M, S> FusedIterator for BKFuzzy<'_, K, V, M, S> where Self: Iterator {}
