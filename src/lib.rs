#![no_std]
extern crate alloc;

use alloc::{boxed::Box, vec::Vec};
use core::{iter::FusedIterator, marker::PhantomData};
use num_traits::{AsPrimitive, SaturatingAdd, SaturatingSub, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize};

pub trait Distance: Copy + PartialOrd + SaturatingAdd + SaturatingSub + Zero {}
impl<T: Copy + PartialOrd + SaturatingAdd + SaturatingSub + Zero> Distance for T {}

pub trait BuildMetric {
    type Distance: Distance;
    type Metric;

    fn build(&self) -> Self::Metric;
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Default)]
pub struct Levenshtein<E, D = usize>(PhantomData<(E, D)>);

impl<E, D: Distance> BuildMetric for Levenshtein<E, D> {
    type Distance = D;
    type Metric = LevenshteinMetric<E, D>;

    fn build(&self) -> Self::Metric {
        Self::Metric {
            cache: Vec::new(),
            _phantom: PhantomData,
        }
    }
}

pub trait Metric<A, B> {
    type Distance: Distance;

    fn distance(&mut self, a: A, b: B) -> Self::Distance;
}

#[derive(Debug)]
pub struct LevenshteinMetric<E, D> {
    cache: Vec<usize>,
    _phantom: PhantomData<(E, D)>,
}

impl<A: AsRef<[E]>, B: AsRef<[E]>, E: PartialEq, D: Distance + 'static> Metric<A, B>
    for LevenshteinMetric<E, D>
where
    usize: AsPrimitive<D>,
{
    type Distance = D;

    fn distance(&mut self, a: A, b: B) -> D {
        let mut a = a.as_ref();
        let mut b = b.as_ref();

        if a.len() < b.len() {
            (a, b) = (b, a);
        }

        while !b.is_empty() && a.first() == b.first() {
            a = &a[1..];
            b = &b[1..];
        }

        while !b.is_empty() && a.last() == b.last() {
            a = &a[..a.len() - 1];
            b = &b[..b.len() - 1];
        }

        if b.is_empty() {
            return a.len().as_();
        }

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

        result.as_()
    }
}

#[cfg_attr(feature = "serde", derive(Serialize))]
#[derive(Clone, Debug)]
pub struct BKMap<K, V, M: BuildMetric> {
    #[cfg_attr(
        feature = "serde",
        serde(bound(serialize = "BKNode<K, V, M::Distance>: Serialize"))
    )]
    root: Option<BKNode<K, V, M::Distance>>,
    build_metric: M,
    #[cfg_attr(feature = "serde", serde(skip))]
    metric: M::Metric,
}

#[cfg(feature = "serde")]
impl<'de, K, V, M: Deserialize<'de> + BuildMetric> Deserialize<'de> for BKMap<K, V, M>
where
    BKNode<K, V, M::Distance>: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct BKMap<K, V, M: BuildMetric> {
            #[serde(bound(deserialize = "BKNode<K, V, M::Distance>: Deserialize<'de>"))]
            root: Option<BKNode<K, V, M::Distance>>,
            build_metric: M,
        }

        let BKMap { root, build_metric } = BKMap::<K, V, M>::deserialize(deserializer)?;
        let metric = build_metric.build();

        Ok(Self {
            root,
            build_metric,
            metric,
        })
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug)]
struct BKNode<K, V, D> {
    dist: D,
    key: K,
    value: V,
    children: Vec<Self>,
}

impl<K, V, D: Distance> BKNode<K, V, D> {
    fn freeze(self) -> FrozenBKNode<K, V, D> {
        FrozenBKNode {
            dist: self.dist,
            key: self.key,
            value: self.value,
            children: self.children.into_iter().map(Self::freeze).collect(),
        }
    }

    fn children_around(&self, dist: D, radius: D) -> impl Iterator<Item = &Self> {
        self.children
            .iter()
            .skip_while(move |child| child.dist < dist.saturating_sub(&radius))
            .take_while(move |child| child.dist <= dist.saturating_add(&radius))
    }

    fn fold<'a, B, Q>(
        &'a self,
        metric: &mut impl for<'b> Metric<&'b Q, &'b K, Distance = D>,
        query: &Q,
        radius: D,
        acc: B,
        f: &mut impl FnMut(B, (D, &'a K, &'a V)) -> B,
    ) -> B {
        let dist = metric.distance(query, &self.key);
        self.children_around(dist, radius).fold(
            if dist <= radius {
                f(acc, (dist, &self.key, &self.value))
            } else {
                acc
            },
            |acc, child| child.fold(metric, query, radius, acc, f),
        )
    }
}

impl<K, V, M: BuildMetric + Default> Default for BKMap<K, V, M> {
    fn default() -> Self {
        Self::with_metric(M::default())
    }
}

impl<K, V, M: BuildMetric> BKMap<K, V, M> {
    #[must_use]
    pub fn with_metric(build_metric: M) -> Self {
        let metric = build_metric.build();
        Self {
            root: None,
            build_metric,
            metric,
        }
    }
}

impl<K, V, M: BuildMetric> BKMap<K, V, M> {
    pub fn insert(&mut self, key: K, value: V)
    where
        M::Metric: for<'a> Metric<&'a K, &'a K, Distance = M::Distance>,
    {
        self.insert_or_modify(key, value, |old, new| *old = new);
    }

    pub fn insert_or_modify(&mut self, key: K, value: V, modify: impl FnOnce(&mut V, V))
    where
        M::Metric: for<'a> Metric<&'a K, &'a K, Distance = M::Distance>,
    {
        self.insert_and_modify(key, value, |old, new| {
            if let Some(new) = new {
                modify(old, new);
            }
        });
    }

    pub fn insert_and_modify(
        &mut self,
        key: K,
        mut value: V,
        modify: impl FnOnce(&mut V, Option<V>),
    ) where
        M::Metric: for<'a> Metric<&'a K, &'a K, Distance = M::Distance>,
    {
        let Some(mut node) = self.root.as_mut() else {
            modify(&mut value, None);
            return self.root = Some(BKNode {
                dist: M::Distance::zero(),
                key,
                value,
                children: Vec::new(),
            });
        };

        loop {
            let dist = self.metric.distance(&key, &node.key);
            if dist.is_zero() {
                return modify(&mut node.value, Some(value));
            }

            let child = node.children.iter().position(|child| child.dist >= dist);

            let Some(child) = child.filter(|&child| node.children[child].dist == dist) else {
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

    pub fn freeze(self) -> FrozenBKMap<K, V, M> {
        FrozenBKMap {
            root: self.root.map(BKNode::freeze),
            build_metric: self.build_metric,
        }
    }

    pub fn fuzzy_search<Q>(&self, query: Q, dist: M::Distance) -> BKFuzzySearch<'_, K, V, M, Q> {
        BKFuzzySearch {
            stack: self.root.as_ref().into_iter().collect(),
            metric: self.build_metric.build(),
            query,
            dist,
        }
    }
}

#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct BKFuzzySearch<'a, K, V, M: BuildMetric, Q> {
    stack: Vec<&'a BKNode<K, V, M::Distance>>,
    metric: M::Metric,
    query: Q,
    dist: M::Distance,
}

impl<'a, K, V, M: BuildMetric<Metric: for<'b> Metric<&'b Q, &'b K, Distance = M::Distance>>, Q>
    Iterator for BKFuzzySearch<'a, K, V, M, Q>
{
    type Item = (M::Distance, &'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.stack.pop()?;
            let dist = self.metric.distance(&self.query, &node.key);

            self.stack.extend(node.children_around(dist, self.dist));

            if dist <= self.dist {
                return Some((dist, &node.key, &node.value));
            }
        }
    }

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.stack.into_iter().fold(init, |acc, node| {
            node.fold(&mut self.metric, &self.query, self.dist, acc, &mut f)
        })
    }
}

impl<K, V, M: BuildMetric, Q> FusedIterator for BKFuzzySearch<'_, K, V, M, Q> where Self: Iterator {}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug)]
pub struct FrozenBKMap<K, V, M: BuildMetric> {
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            serialize = "FrozenBKNode<K, V, M::Distance>: Serialize",
            deserialize = "FrozenBKNode<K, V, M::Distance>: Deserialize<'de>"
        ))
    )]
    root: Option<FrozenBKNode<K, V, M::Distance>>,
    build_metric: M,
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug)]
struct FrozenBKNode<K, V, D> {
    dist: D,
    key: K,
    value: V,
    children: Box<[Self]>,
}

impl<K, V, D: Distance> FrozenBKNode<K, V, D> {
    fn children_around(&self, dist: D, radius: D) -> impl Iterator<Item = &Self> {
        self.children
            .iter()
            .skip_while(move |child| child.dist < dist.saturating_sub(&radius))
            .take_while(move |child| child.dist <= dist.saturating_add(&radius))
    }

    fn fold<'a, B, Q>(
        &'a self,
        metric: &mut impl for<'b> Metric<&'b Q, &'b K, Distance = D>,
        query: &Q,
        radius: D,
        acc: B,
        f: &mut impl FnMut(B, (D, &'a K, &'a V)) -> B,
    ) -> B {
        let dist = metric.distance(query, &self.key);
        self.children_around(dist, radius).fold(
            if dist <= radius {
                f(acc, (dist, &self.key, &self.value))
            } else {
                acc
            },
            |acc, child| child.fold(metric, query, radius, acc, f),
        )
    }
}

impl<K, V, M: BuildMetric> FrozenBKMap<K, V, M> {
    pub fn fuzzy_search<Q>(
        &self,
        query: Q,
        dist: M::Distance,
    ) -> FrozenBKFuzzySearch<'_, K, V, M, Q> {
        FrozenBKFuzzySearch {
            stack: self.root.as_ref().into_iter().collect(),
            metric: self.build_metric.build(),
            query,
            dist,
        }
    }
}

#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct FrozenBKFuzzySearch<'a, K, V, M: BuildMetric, Q> {
    stack: Vec<&'a FrozenBKNode<K, V, M::Distance>>,
    metric: M::Metric,
    query: Q,
    dist: M::Distance,
}

impl<'a, K, V, M: BuildMetric<Metric: for<'b> Metric<&'b Q, &'b K, Distance = M::Distance>>, Q>
    Iterator for FrozenBKFuzzySearch<'a, K, V, M, Q>
{
    type Item = (M::Distance, &'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.stack.pop()?;
            let dist = self.metric.distance(&self.query, &node.key);

            self.stack.extend(node.children_around(dist, self.dist));

            if dist <= self.dist {
                return Some((dist, &node.key, &node.value));
            }
        }
    }

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.stack.into_iter().fold(init, |acc, node| {
            node.fold(&mut self.metric, &self.query, self.dist, acc, &mut f)
        })
    }
}

impl<K, V, M: BuildMetric, Q> FusedIterator for FrozenBKFuzzySearch<'_, K, V, M, Q> where
    Self: Iterator
{
}
