use std::{
    cmp::Ordering,
    fmt,
    marker::PhantomData,
    ops::{Index, IndexMut},
};

/// The Pool and Bag collections are inspired by Joren Brunekreef's work.
/// See for example https://github.com/JorenB/2d-cdt
///
/// The Link-Cut Tree implementation is based on an MIT lecture by Erik Demaine.

/// Label that is used for indexing into collections.
pub struct Label<T> {
    value: usize,
    object_type: std::marker::PhantomData<T>,
}

/// Collection with stable indices and O(1) time insert and remove operations. Can be indexed using `Label`.
#[derive(Debug, Clone)]
pub struct Pool<T> {
    elements: Box<[Element<T>]>,
    current_hole: usize,
    size: usize,
}

/// Iterator for the `Pool` collection.
#[derive(Debug)]
pub struct PoolIter<'a, T> {
    pool: &'a Pool<T>,
    index: usize,
}

/// Collection of `Labels` referring to some `Pool`. Can be sampled from in O(1) time.
#[derive(Debug, Clone)]
pub struct Bag<T> {
    indices: Box<[Option<usize>]>,
    labels: Box<[Option<Label<T>>]>,
    size: usize,
}

/// Represents a `Node` in a Link-Cut tree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node<T> {
    value: T,
    parent: Parent<T>,
    children: Children<T>,
    d_flip: bool,
    d_depth: usize,
}

impl<T> Node<T> {
    pub fn new(value: T) -> Self {
        Node {
            value,
            parent: Parent::Root,
            children: Children(None, None),
            d_flip: false,
            d_depth: 0,
        }
    }
}

struct Children<T>(Option<NodeLabel<T>>, Option<NodeLabel<T>>);

#[derive(Clone, Debug)]
struct NaiveForest {
    nodes: Vec<Vec<usize>>,
}

#[derive(Debug, PartialEq, Clone)]
enum Element<T> {
    Object(T),
    Hole(usize),
}

enum Parent<T> {
    BinTree { parent: NodeLabel<T>, side: bool },
    Path { parent: NodeLabel<T> },
    Root,
}

impl<T> Label<T> {
    fn new(value: usize) -> Label<T> {
        Label {
            value,
            object_type: PhantomData,
        }
    }

    #[cfg(test)]
    pub fn from_value(value: usize) -> Label<T> {
        Label {
            value,
            object_type: PhantomData,
        }
    }
}

impl<T> Pool<T> {
    /// Construct a `Pool` with a given `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Pool {
            elements: (0..capacity).map(|i| Element::Hole(i + 1)).collect(),
            current_hole: 0,
            size: 0,
        }
    }

    /// Insert an `object`, and return its `Label`.
    pub fn insert(&mut self, object: T) -> Label<T> {
        let label = self.current_hole;
        debug_assert_ne!(label, self.elements.len(), "Pool is full!");
        if let Element::Hole(next) = self.elements[label] {
            self.elements[label] = Element::Object(object);
            self.current_hole = next;
            self.size += 1;
            Label::new(label)
        } else {
            unreachable!("current hole ({label}) should be a hole");
        }
    }

    /// Remove an object with a given `Label`.
    pub fn remove(&mut self, label: Label<T>) {
        let index = label.value;
        debug_assert!(self.contains(label));
        self.elements[index] = Element::Hole(self.current_hole);
        self.current_hole = index;
        self.size -= 1;
    }

    /// Return the next `Label`.
    pub fn next_label(&self) -> Option<Label<T>> {
        if self.current_hole == self.elements.len() {
            None
        } else {
            Some(Label::new(self.current_hole))
        }
    }

    /// Return the total number of objects in the `Pool`.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Return the maximum `capacity` of the `Pool`.
    pub fn capacity(&self) -> usize {
        self.elements.len()
    }

    /// Check if the `Pool` contains an object with a given `Label`.
    pub fn contains(&self, label: Label<T>) -> bool {
        match self.elements[label.value] {
            Element::Object(_) => true,
            Element::Hole(_) => false,
        }
    }
}

type NodeLabel<T> = Label<Node<T>>;

pub trait LinkCutTree<T> {
    fn get_node(&self, label: NodeLabel<T>) -> &Node<T>;
    fn get_node_mut(&mut self, label: NodeLabel<T>) -> &mut Node<T>;

    /// Find the root of the tree containing the node `u`.
    fn find_root(&mut self, u: NodeLabel<T>) -> NodeLabel<T> {
        // loop through the splay tree to find the root
        let mut x = u;
        let mut flip = self.get_node(u).d_flip;
        while let Some(y) = self.get_node(x).children[flip] {
            x = y;
            flip ^= self.get_node(x).d_flip;
        }

        self.splay(x); // this is required for some reason

        x
    }

    /// Cut a node `u` from its parent. If `u` has no parent, the LCT is not edited.
    fn cut(&mut self, u: NodeLabel<T>) {
        // select the proper child based on d_flip
        let d_flip = self.get_node(u).d_flip;
        if let Some(p) = self.get_node(u).children[d_flip] {
            // make the cut
            self.get_node_mut(p).parent = Parent::Root;
            self.get_node_mut(u).children[d_flip] = None;

            // update flip and depth values
            self.get_node_mut(p).d_flip ^= d_flip;
            self.get_node_mut(p).d_depth = self.get_node(u).d_depth - self.get_node(p).d_depth;
            self.get_node_mut(u).d_depth = 0;
        }
    }

    /// Link a parent node `p` with a child node `c`. Here `c` is assumed to be the root of its tree.
    fn link(&mut self, p: NodeLabel<T>, c: NodeLabel<T>) {
        debug_assert_eq!(self.find_root(c), c);
        debug_assert_ne!(self.find_root(p), c);
        debug_assert_ne!(p, c);

        // link the nodes
        self.get_node_mut(c).d_flip = false;
        self.get_node_mut(c).children[false] = Some(p);
        self.get_node_mut(p).parent = Parent::BinTree {
            parent: c,
            side: false,
        };

        // update depth values
        self.get_node_mut(c).d_depth = self.get_node(p).d_depth + 1;
        self.get_node_mut(p).d_depth = 1;
    }

    /// Make `u` the root of its tree.
    fn evert(&mut self, u: NodeLabel<T>) {
        // evert the node
        self.get_node_mut(u).d_flip = !self.get_node(u).d_flip;
        self.get_node_mut(u).d_depth = 0;
    }

    /// Find the distance between `u` and the root of its tree.
    /// Here distance is defined as the number of links between two nodes in a tree.
    fn depth(&self, u: NodeLabel<T>) -> usize {
        // walk up the tree until the root is reached
        let mut x = u;
        let mut depth: isize = 0;
        loop {
            let d_depth = self.get_node(x).d_depth as isize;

            // update or return the depth as required
            let p = match self.get_node(x).parent {
                Parent::BinTree { parent, side } => {
                    if side {
                        depth += d_depth;
                    } else {
                        depth -= d_depth;
                    };
                    parent
                }
                Parent::Path { parent: _ } => unreachable!("State is corrupted."),
                Parent::Root => {
                    let d_depth = self.get_node(x).d_depth as isize;
                    return (depth + d_depth).unsigned_abs();
                }
            };

            // flip the depth if required
            if self.get_node(p).d_flip {
                depth = -depth;
            }

            x = p;
        }
    }

    /// Return the `i`th node on the path from the root to `u`.
    fn index_depth(&self, u: NodeLabel<T>, i: usize) -> NodeLabel<T> {
        debug_assert!(i <= self.depth(u), "i = {}, depth = {}", i, self.depth(u));

        // walk down the tree until the ith node is reached
        let mut label = u;
        let mut flip = self.get_node(u).d_flip;
        let mut depth = self.get_node(u).d_depth;
        loop {
            // find which side to walk down to or return if depth is reached
            let side = match i.cmp(&depth) {
                Ordering::Less => flip,
                Ordering::Equal => return label,
                Ordering::Greater => !flip,
            };

            // calculate the depth of the next node down
            label = self.get_node(label).children[side].unwrap();
            if flip ^ side {
                depth += self.get_node(label).d_depth;
            } else {
                depth -= self.get_node(label).d_depth;
            }

            // update flip value
            flip ^= self.get_node(label).d_flip;
        }
    }

    /// Get a mutable reference to the value of node `u`.
    fn value_mut(&mut self, u: NodeLabel<T>) -> &mut T {
        &mut self.get_node_mut(u).value
    }

    /// Get an immutable reference to the value of node `u`.
    fn value(&self, u: NodeLabel<T>) -> &T {
        &self.get_node(u).value
    }

    /// Bring `u` in the same splay tree as the root of its real tree, and make `u` the root of its splay tree.
    fn expose(&mut self, u: NodeLabel<T>) {
        self.splice(u, None);

        // keep splicing until u is in the same splay tree as the root
        while let Parent::Path { parent: p } = self.get_node(u).parent {
            let u_side = self.splice(p, Some(u));
            self.rotate(u, p, u_side);
        }
    }

    /// Replace the preferred child of `p` by `u_opt`. Return the side at which u is placed as a child of `p`.
    fn splice(&mut self, p: NodeLabel<T>, u_opt: Option<NodeLabel<T>>) -> bool {
        self.splay(p);

        let d_flip = self.get_node(p).d_flip;
        let u_side = !d_flip;

        // connect the current preferred child as a path child
        if let Some(x) = self.get_node(p).children[u_side] {
            let mut x_node = &mut self.get_node_mut(x);
            x_node.parent = Parent::Path { parent: p };
            x_node.d_flip ^= d_flip;
        }

        // make u the new preferred child (if it exists)
        if let Some(u) = u_opt {
            let mut u_node = self.get_node_mut(u);
            u_node.parent = Parent::BinTree {
                parent: p,
                side: u_side,
            };
            u_node.d_flip ^= d_flip;
        }
        self.get_node_mut(p).children[u_side] = u_opt;
        u_side
    }

    /// Make `u` the root of its splay tree. Do this in such a way, so that the splay tree becomes balanced.
    fn splay(&mut self, u: NodeLabel<T>) {
        while let Parent::BinTree {
            parent: p,
            side: u_side,
        } = self.get_node(u).parent
        {
            if let Parent::BinTree {
                parent: g,
                side: p_side,
            } = self.get_node(p).parent
            {
                let flip = self.get_node(p).d_flip;
                if flip == (u_side == p_side) {
                    self.rotate(u, p, u_side);
                    self.rotate(u, g, !u_side ^ flip);
                } else {
                    self.rotate(p, g, p_side);
                    self.rotate(u, p, p_side ^ flip);
                };
            } else {
                self.rotate(u, p, u_side);
            }
        }
    }

    /// Move `u` up one level in the splay tree.
    fn rotate(&mut self, u: NodeLabel<T>, p: NodeLabel<T>, u_side: bool) {
        let Node {
            value: _,
            parent: _,
            children: u_children,
            d_flip: u_d_flip,
            d_depth: u_d_depth,
        } = *self.get_node(u);

        let Node {
            value: _,
            parent: p_parent,
            children: _,
            d_flip: p_d_flip,
            d_depth: p_d_depth,
        } = *self.get_node(p);

        let x_side = u_d_flip == u_side;
        let x_opt = u_children[x_side];
        let rel_g_side = match p_parent {
            Parent::BinTree { parent: g, side } => {
                self.get_node_mut(g).children[side] = Some(u);
                u_side == side
            }
            _ => u_side,
        };

        // update node
        let mut u_node = self.get_node_mut(u);
        u_node.parent = p_parent;
        u_node.children[x_side] = Some(p);
        u_node.d_flip ^= p_d_flip;
        u_node.d_depth = if p_d_flip ^ rel_g_side {
            u_d_depth + p_d_depth
        } else {
            u_d_depth.abs_diff(p_d_depth)
        };

        // update parent
        let mut p_node = self.get_node_mut(p);
        p_node.parent = Parent::BinTree {
            parent: u,
            side: x_side,
        };
        p_node.children[u_side] = x_opt;
        p_node.d_flip = u_d_flip;
        p_node.d_depth = u_d_depth;

        // move the middle subtree to its new parent
        if let Some(x) = x_opt {
            let mut x_node = self.get_node_mut(x);
            x_node.d_depth = x_node.d_depth.abs_diff(u_d_depth);
            x_node.d_flip ^= u_d_flip;
            x_node.parent = Parent::BinTree {
                parent: p,
                side: u_side,
            };
        }
    }

    /*     #[cfg(test)]
     fn sanity_check(&self) {
        for label in self {
            // check parent reciprocrity
            match self.get_node(label).parent {
                Parent::Root => (),
                Parent::Path { parent: _ } => (),
                Parent::BinTree { parent, side } => match side {
                    false => assert_eq!(self[parent].children.0, Some(label)),
                    true => assert_eq!(self[parent].children.1, Some(label)),
                },
            }

            // check child reciprocrity
            if let Some(c) = self[label].children.0 {
                assert_eq!(
                    self[c].parent,
                    Parent::BinTree {
                        parent: label,
                        side: false
                    }
                );
            }
            if let Some(c) = self[label].children.1 {
                assert_eq!(
                    self[c].parent,
                    Parent::BinTree {
                        parent: label,
                        side: true
                    }
                );
            }
        }
    } */
}

impl<T> LinkCutTree<T> for Pool<Node<T>> {
    fn get_node(&self, label: NodeLabel<T>) -> &Node<T> {
        &self[label]
    }
    fn get_node_mut(&mut self, label: NodeLabel<T>) -> &mut Node<T> {
        &mut self[label]
    }
}

impl<T> Bag<T> {
    /// Construct a `Bag` with a given `capacity`.
    pub fn with_capacity(capacity: usize) -> Bag<T> {
        Bag {
            indices: (0..capacity).map(|_| None).collect(),
            labels: (0..capacity).map(|_| None).collect(),
            size: 0,
        }
    }

    /// Insert a `Label`.
    pub fn insert(&mut self, label: Label<T>) {
        debug_assert_eq!(
            self.indices[label.value], None,
            "label {} is already in bag",
            label.value
        );

        self.indices[label.value] = Some(self.size);
        self.labels[self.size] = Some(label);
        self.size += 1;
    }

    /// Remove a `Label`.
    pub fn remove(&mut self, label: Label<T>) {
        self.size -= 1;
        let index = self.indices[label.value].expect("index should be in bag");
        self.labels[index] = self.labels[self.size];
        self.indices[self.labels[self.size]
            .expect("label should be in bag")
            .value] = Some(index);
        self.indices[label.value] = None;
        self.labels[self.size] = None;
    }

    /// Randomly sample a `Label` from the `Bag`.
    pub fn sample(&self) -> Label<T> {
        let index = fastrand::usize(0..(self.size));
        self.labels[index].expect("label at listed index should be in bag")
    }

    /// Check if the `Bag` contains a given `Label`.
    pub fn contains(&self, label: Label<T>) -> bool {
        self.indices[label.value].is_some()
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl<'a, T> IntoIterator for &'a Pool<T> {
    type Item = Label<T>;
    type IntoIter = PoolIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        PoolIter {
            pool: self,
            index: 0,
        }
    }
}

impl<'a, T> Iterator for PoolIter<'a, T> {
    type Item = Label<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let size = self.pool.elements.len();
        for i in self.index..size {
            match self.pool.elements[i] {
                Element::Object(_) => {
                    self.index = i + 1;
                    return Some(Label::new(i));
                }
                Element::Hole(_) => (),
            };
        }
        None
    }
}

impl<T> Index<Label<T>> for Pool<T> {
    type Output = T;

    fn index(&self, label: Label<T>) -> &Self::Output {
        match &self.elements[label.value] {
            Element::Object(object) => object,
            Element::Hole(_) => panic!("Label {} is not in use!", label.value),
        }
    }
}

impl<T> Index<bool> for Children<T> {
    type Output = Option<NodeLabel<T>>;

    fn index(&self, side: bool) -> &Self::Output {
        if side {
            &self.1
        } else {
            &self.0
        }
    }
}

impl<T> IndexMut<Label<T>> for Pool<T> {
    fn index_mut(&mut self, label: Label<T>) -> &mut Self::Output {
        match &mut self.elements[label.value] {
            Element::Object(object) => object,
            Element::Hole(_) => panic!("Label {} is not in use!", label.value),
        }
    }
}

impl<T> IndexMut<bool> for Children<T> {
    fn index_mut(&mut self, side: bool) -> &mut Self::Output {
        if side {
            &mut self.1
        } else {
            &mut self.0
        }
    }
}

impl<T> Copy for Label<T> {}

impl<T> Copy for Parent<T> {}

impl<T> Copy for Children<T> {}

impl<T> Clone for Label<T> {
    fn clone(&self) -> Label<T> {
        *self
    }
}

impl<T> Clone for Parent<T> {
    fn clone(&self) -> Parent<T> {
        *self
    }
}

impl<T> Clone for Children<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> fmt::Debug for Label<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Label").field("value", &self.value).finish()
    }
}

impl<T> fmt::Debug for Parent<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Parent::Path { parent } => f
                .debug_struct("Parent::Path")
                .field("parent", parent)
                .finish(),
            Parent::BinTree { parent, side } => f
                .debug_struct("Parent::BinTree")
                .field("parent", parent)
                .field("side", side)
                .finish(),
            Parent::Root => f.debug_struct("Parent::Root").finish(),
        }
    }
}

impl<T> fmt::Debug for Children<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Children")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

impl<T> fmt::Display for Label<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<T: fmt::Display> fmt::Display for Pool<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.elements
            .iter()
            .enumerate()
            .try_for_each(|(i, element)| match element {
                Element::Hole(_) => Ok(()),
                Element::Object(obj) => {
                    writeln!(f, "[{i}]:\t{obj}")
                }
            })?;
        Ok(())
    }
}

impl<T: fmt::Display> fmt::Display for Bag<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.labels.iter().try_for_each(|element| match element {
            None => Ok(()),
            Some(label) => {
                write!(f, "{label}, ")
            }
        })?;
        writeln!(f)?;
        Ok(())
    }
}

impl<T: Copy> fmt::Display for Pool<Node<T>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let forest = NaiveForest::from(self);
        write!(f, "{forest}")
    }
}

impl fmt::Display for NaiveForest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, _) in self.nodes.iter().enumerate() {
            if !self.nodes.iter().any(|children| children.contains(&i)) {
                self.print_children(f, i, 0)?;
            }
        }
        Ok(())
    }
}

impl NaiveForest {
    fn print_children(&self, f: &mut fmt::Formatter<'_>, node: usize, depth: usize) -> fmt::Result {
        if depth > 0 {
            for _ in 0..(depth - 1) {
                write!(f, "│")?;
            }
            writeln!(f, "├{node}")?;
        } else {
            writeln!(f, "{node}")?;
        }
        for child in self.nodes[node].iter() {
            self.print_children(f, *child, depth + 1)?;
        }
        Ok(())
    }
}

impl<T> PartialEq for Label<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T> PartialEq for Parent<T> {
    fn eq(&self, other: &Self) -> bool {
        match *self {
            Parent::BinTree {
                parent: parent1,
                side: side1,
            } => match *other {
                Parent::BinTree {
                    parent: parent2,
                    side: side2,
                } => parent1 == parent2 && side1 == side2,
                _ => false,
            },
            Parent::Path { parent: parent1 } => match *other {
                Parent::Path { parent: parent2 } => parent1 == parent2,
                _ => false,
            },
            Parent::Root => matches!(*other, Parent::Root),
        }
    }
}

impl<T> PartialEq for Children<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<T> Eq for Label<T> {}

impl<T> PartialOrd for Label<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.value.cmp(&other.value))
    }
}

impl<T> Ord for Label<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T: Clone> From<&Pool<Node<T>>> for NaiveForest {
    fn from(value: &Pool<Node<T>>) -> Self {
        let mut forest = value.clone();
        let labels = value.into_iter().collect::<Vec<_>>();
        let mut nodes = vec![Vec::<usize>::new(); value.capacity()];
        for label in labels {
            forest.expose(label);
            let depth = forest.depth(label);
            if depth > 0 {
                let parent = forest.index_depth(label, depth - 1);
                nodes[parent.value].push(label.value);
            }
        }
        NaiveForest { nodes }
    }
}
/*
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct Foo;

    #[test]
    fn it_works() {
        // create pool with capacity 1
        let mut pool = Pool::with_capacity(1);
        assert_eq!(pool.elements[0], Element::Hole(1));
        assert_eq!(pool.current_hole, 0);
        assert_eq!(pool.size, 0);

        // create bag with capacity 1
        let mut bag = Bag::with_capacity(1);
        assert_eq!(bag.indices[0], None);
        assert_eq!(bag.labels[0], None);
        assert_eq!(bag.size, 0);

        // insert object into pool
        let label = pool.insert(Foo);
        assert_eq!(pool.elements[0], Element::Object(Foo));
        assert_eq!(pool.current_hole, 1);
        assert_eq!(pool.size, 1);
        assert!(pool.contains(label));

        // insert label into bag
        bag.insert(label);
        assert_eq!(bag.indices[0], Some(0));
        assert_eq!(bag.labels[0], Some(label));
        assert_eq!(bag.size, 1);
        assert!(bag.contains(label));

        // remove label from bag
        bag.remove(label);
        assert_eq!(bag.indices[0], None);
        assert_eq!(bag.labels[0], None);
        assert_eq!(bag.size, 0);
        assert!(!bag.contains(label));

        // remove object from pool
        pool.remove(label);
        assert_eq!(pool.elements[0], Element::Hole(1));
        assert_eq!(pool.current_hole, 0);
        assert_eq!(pool.size, 0);
        assert!(!pool.contains(label));
    }

    #[test]
    fn insert_remove() {
        // create pool and bag
        let mut pool = Pool::with_capacity(3);
        let mut bag = Bag::with_capacity(3);

        // insert 3 objects into pool
        let label1 = pool.insert(Foo);
        let label2 = pool.insert(Foo);
        let label3 = pool.insert(Foo);

        // insert 3 labels into bag
        bag.insert(label1);
        bag.insert(label2);
        bag.insert(label3);

        // remove second label from bag
        bag.remove(label2);
        assert_eq!(bag.indices[0], Some(0));
        assert_eq!(bag.indices[1], None);
        assert_eq!(bag.indices[2], Some(1));
        assert_eq!(bag.labels[0], Some(label1));
        assert_eq!(bag.labels[1], Some(label3));
        assert_eq!(bag.labels[2], None);
        assert_eq!(bag.size, 2);
        assert!(bag.contains(label1));
        assert!(!bag.contains(label2));
        assert!(bag.contains(label3));

        // remove second object from pool
        pool.remove(label2);
        assert_eq!(pool.elements[1], Element::Hole(3));
        assert_eq!(pool.current_hole, 1);
        assert_eq!(pool.size, 2);
        assert!(pool.contains(label1));
        assert!(!pool.contains(label2));
        assert!(pool.contains(label3));
    }

    #[test]
    #[should_panic]
    fn pool_cap() {
        let mut pool = Pool::with_capacity(1);
        let _ = pool.insert(Foo);
        let _ = pool.insert(Foo);
    }

    #[test]
    #[should_panic]
    fn bag_cap() {
        let mut pool = Pool::with_capacity(2);
        let mut bag = Bag::with_capacity(1);
        let label1 = pool.insert(Foo);
        let label2 = pool.insert(Foo);
        bag.insert(label1);
        bag.insert(label2);
    }

    #[test]
    fn pool_iter() {
        let mut pool = Pool::with_capacity(3);
        let label1 = pool.insert(Foo);
        let label2 = pool.insert(Foo);
        let label3 = pool.insert(Foo);
        pool.remove(label2);

        let mut pool_iter = pool.into_iter();
        assert_eq!(pool_iter.next(), Some(label1));
        assert_eq!(pool_iter.next(), Some(label3));
        assert_eq!(pool_iter.next(), None);
    }

    #[test]
    fn tree_works() {
        let mut lct = Pool::with_capacity(1);
        let v = lct.add_node(());
        lct.splay(v);
        lct.expose(v);
    }

    #[test]
    fn rotate_basic() {
        let mut lct = Pool::with_capacity(2);

        // define nodes
        let o = lct.add_node("node");
        let p = lct.add_node("parent");

        // link nodes
        lct[p].children[false] = Some(o);
        lct[o].parent = Parent::BinTree {
            parent: p,
            side: false,
        };

        // rotate
        lct.rotate(o, p, false);

        // check children
        assert_eq!(lct[o].children, Children(None, Some(p)));
        assert_eq!(lct[p].children, Children(None, None));

        // check parents
        assert_eq!(lct[o].parent, Parent::Root);
        assert_eq!(
            lct[p].parent,
            Parent::BinTree {
                parent: o,
                side: true
            }
        );
    }

    #[test]
    fn rotate_attachments() {
        let mut lct = Pool::with_capacity(7);

        // define nodes
        let o = lct.add_node("node");
        let p = lct.add_node("parent");
        let g = lct.add_node("grandparent");
        let a = lct.add_node("a");
        let b = lct.add_node("b");
        let c = lct.add_node("c");
        let d = lct.add_node("d");

        // link nodes
        lct[o].children = Children(Some(a), Some(b));
        lct[p].children = Children(Some(o), Some(c));
        lct[g].children = Children(Some(p), Some(d));
        lct[o].parent = Parent::BinTree {
            parent: p,
            side: false,
        };
        lct[p].parent = Parent::BinTree {
            parent: g,
            side: false,
        };
        lct[a].parent = Parent::BinTree {
            parent: o,
            side: false,
        };
        lct[b].parent = Parent::BinTree {
            parent: o,
            side: true,
        };
        lct[c].parent = Parent::BinTree {
            parent: p,
            side: true,
        };
        lct[d].parent = Parent::BinTree {
            parent: g,
            side: true,
        };

        // rotate
        lct.rotate(o, p, false);
        dbg!(&lct);

        // check children
        assert_eq!(lct[o].children, Children(Some(a), Some(p)));
        assert_eq!(lct[p].children, Children(Some(b), Some(c)));
        assert_eq!(lct[g].children, Children(Some(o), Some(d)));
        assert_eq!(lct[a].children, Children(None, None));
        assert_eq!(lct[b].children, Children(None, None));
        assert_eq!(lct[c].children, Children(None, None));
        assert_eq!(lct[d].children, Children(None, None));

        // check parents
        assert_eq!(
            lct[o].parent,
            Parent::BinTree {
                parent: g,
                side: false
            }
        );
        assert_eq!(
            lct[p].parent,
            Parent::BinTree {
                parent: o,
                side: true
            }
        );
        assert_eq!(lct[g].parent, Parent::Root);
        assert_eq!(
            lct[a].parent,
            Parent::BinTree {
                parent: o,
                side: false
            }
        );
        assert_eq!(
            lct[b].parent,
            Parent::BinTree {
                parent: p,
                side: false
            }
        );
        assert_eq!(
            lct[c].parent,
            Parent::BinTree {
                parent: p,
                side: true
            }
        );
        assert_eq!(
            lct[d].parent,
            Parent::BinTree {
                parent: g,
                side: true
            }
        );
    }

    #[test]
    fn expose_line() {
        let mut lct = Pool::with_capacity(5);

        // define nodes
        let n0 = lct.add_node("0");
        let n1 = lct.add_node("1");
        let n2 = lct.add_node("2");
        let n3 = lct.add_node("3");
        let n4 = lct.add_node("4");

        // link nodes
        lct[n1].children = Children(None, Some(n2));
        lct[n3].children = Children(None, Some(n4));
        lct[n2].parent = Parent::BinTree {
            parent: n1,
            side: true,
        };
        lct[n4].parent = Parent::BinTree {
            parent: n3,
            side: true,
        };
        lct[n1].parent = Parent::Path { parent: n0 };
        lct[n3].parent = Parent::Path { parent: n2 };

        // expose
        lct.expose(n3);
        dbg!(&lct);

        // check parents
        assert_eq!(lct[n3].parent, Parent::Root);
        assert_eq!(lct[n4].parent, Parent::Path { parent: n3 });
    }

    #[test]
    fn find_root() {
        let mut lct = Pool::with_capacity(7);

        // define nodes
        let o = lct.add_node("node");
        let p = lct.add_node("parent");
        let g = lct.add_node("grandparent");
        let a = lct.add_node("a");
        let b = lct.add_node("b");
        let c = lct.add_node("c");
        let d = lct.add_node("d");

        // link nodes
        lct[o].children = Children(Some(a), Some(b));
        lct[p].children = Children(Some(o), Some(c));
        lct[g].children = Children(Some(p), Some(d));
        lct[o].parent = Parent::BinTree {
            parent: p,
            side: false,
        };
        lct[p].parent = Parent::BinTree {
            parent: g,
            side: false,
        };
        lct[a].parent = Parent::BinTree {
            parent: o,
            side: false,
        };
        lct[b].parent = Parent::BinTree {
            parent: o,
            side: true,
        };
        lct[c].parent = Parent::BinTree {
            parent: p,
            side: true,
        };
        lct[d].parent = Parent::BinTree {
            parent: g,
            side: true,
        };

        // find root
        assert_eq!(lct.find_root(g), a);
    }

    #[test]
    fn link_cut() {
        let mut lct = Pool::with_capacity(2);

        // define nodes
        let a = lct.add_node("a");
        let b = lct.add_node("b");

        // link nodes
        lct.link(a, b);

        // cut nodes
        lct.cut(b);
    }

    #[test]
    fn evert_find_root() {
        let mut lct = Pool::with_capacity(5);

        // define nodes
        let n0 = lct.add_node("0");
        let n1 = lct.add_node("1");
        let n2 = lct.add_node("2");
        let n3 = lct.add_node("3");
        let n4 = lct.add_node("4");

        // link nodes
        lct.link(n1, n0);
        lct.link(n2, n1);
        lct.link(n3, n2);
        lct.link(n4, n3);

        // evert
        lct.evert(n2);

        // check
        assert_eq!(lct.find_root(n0), n2);
        assert_eq!(lct.find_root(n1), n2);
        assert_eq!(lct.find_root(n2), n2);
        assert_eq!(lct.find_root(n3), n2);
        assert_eq!(lct.find_root(n4), n2);
    }

    #[test]
    fn basic_depth() {
        // generate a basic lct
        let size = 3;
        let (mut lct, labels) = linear_lct(size);

        // check initial depth
        labels
            .iter()
            .enumerate()
            .for_each(|(i, &label)| assert_eq!(lct.depth(label), i));

        // evert middle node and check new depths
        lct.evert(labels[1]);
        dbg!(&lct);
        assert_eq!(lct.depth(labels[0]), 1);
        assert_eq!(lct.depth(labels[1]), 0);
        assert_eq!(lct.depth(labels[2]), 1);
    }

    #[test]
    fn splay_depth() {
        // create lct
        let mut lct = Pool::with_capacity(3);
        let a = lct.add_node("a");
        let b = lct.add_node("b");
        let c = lct.add_node("c");

        // link and splay
        lct.link(a, b);
        lct.link(b, c);
        lct.splay(b);
        lct.splay(a);
        lct.splay(c);

        // check depths
        assert_eq!(lct.depth(a), 0);
        assert_eq!(lct.depth(b), 1);
        assert_eq!(lct.depth(c), 2);
    }

    #[test]
    fn expose_depth() {
        // create lct
        let mut lct = Pool::with_capacity(3);
        let a = lct.add_node("a");
        let b = lct.add_node("b");
        let c = lct.add_node("c");

        // link and expose
        lct.link(a, b);
        lct.link(b, c);
        lct.expose(b);
        lct.expose(a);
        lct.expose(c);

        // check depths
        assert_eq!(lct.depth(a), 0);
        assert_eq!(lct.depth(b), 1);
        assert_eq!(lct.depth(c), 2);
    }

    #[test]
    fn evert_depth() {
        // create lct
        let (mut lct, labels) = linear_lct(5);

        // evert
        lct.evert(labels[2]);

        // check depths
        assert_eq!(lct.depth(labels[0]), 2);
        assert_eq!(lct.depth(labels[1]), 1);
        assert_eq!(lct.depth(labels[2]), 0);
        assert_eq!(lct.depth(labels[3]), 1);
        assert_eq!(lct.depth(labels[4]), 2);
    }

    #[test]
    fn evert_cut_link_depth() {
        // create lct
        let (mut lct, labels) = linear_lct(5);

        // evert
        lct.evert(labels[2]);

        // cut and link
        lct.cut(labels[3]);
        lct.link(labels[0], labels[3]);

        // check depths
        assert_eq!(lct.depth(labels[0]), 2);
        assert_eq!(lct.depth(labels[1]), 1);
        assert_eq!(lct.depth(labels[2]), 0);
        assert_eq!(lct.depth(labels[3]), 3);
        assert_eq!(lct.depth(labels[4]), 4);
    }

    #[test]
    fn random_evert() {
        // generate a basic lct
        let size = 6;
        let (mut lct, labels) = linear_lct(size);

        // randomly evert
        for _ in 0..(size * size * size * size) {
            let root = labels[fastrand::usize(0..size)];
            lct.evert(root);
            let label = labels[fastrand::usize(0..size)];
            lct.expose(label);
            lct.sanity_check();
        }
    }

    #[test]
    fn special_case_depth() {
        // generate a basic lct
        let size = 4;
        let (mut lct, labels) = linear_lct(size);

        // do some operations
        reattach(&mut lct, labels[0], labels[3], 1);
        reattach(&mut lct, labels[2], labels[1], 1);
        lct.evert(labels[3]);
        lct.expose(labels[1]);

        // check the depth
        let depth = lct.depth(labels[1]);
        assert!(depth > 0);
    }

    fn reattach<T: fmt::Debug>(
        lct: &mut Pool<Node<T>>,
        label1: NodeLabel<T>,
        label2: NodeLabel<T>,
        index: usize,
    ) {
        // restructure tree
        lct.evert(label1);
        lct.expose(label2);

        // cut and link back together
        let cut = lct.index_depth(label2, index + 1);
        dbg!(cut);
        lct.cut(cut);
        lct.evert(label2);
        lct.link(label1, label2);
    }

    #[test]
    fn random_cut_link() {
        // generate a basic lct
        let size = 6;
        let (mut lct, labels) = linear_lct(size);

        // randomly cut and link
        for _ in 0..(size * size * size * size) {
            let label1 = labels[fastrand::usize(0..size)];
            let label2 = labels[fastrand::usize(0..size)];
            if label1 != label2 {
                // restructure tree
                lct.evert(label1);

                // find a node to cut
                let depth = lct.depth(label2);
                let index = fastrand::usize(0..depth);
                let cut = lct.index_depth(label2, index + 1);

                // cut and link back together
                lct.cut(cut);
                lct.evert(label2);
                lct.link(label1, label2);
            }
            lct.sanity_check();
        }
    }

    fn linear_lct(size: usize) -> (Pool<Node<usize>>, Vec<Label<Node<usize>>>) {
        let mut lct = Pool::with_capacity(size);
        let labels = (0..size)
            .map(|i| lct.add_node(i))
            .collect::<Vec<Label<Node<usize>>>>();
        for i in 0..(size - 1) {
            let parent = labels[i];
            let child = labels[i + 1];
            lct.link(parent, child);
        }
        (lct, labels)
    }
}
 */
