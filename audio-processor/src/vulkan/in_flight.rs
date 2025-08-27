use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

#[derive(Copy, Clone)]
pub(crate) struct InFlightCounter {
    frames_in_flight: usize,
    counter: usize,
}

impl InFlightCounter {
    pub(crate) fn new(frames_in_flight: usize) -> Self {
        Self {
            frames_in_flight,
            counter: 0
        }
    }

    pub(crate) fn next(&self) -> Self {
        Self {
            counter: (self.counter + 1) % self.frames_in_flight,
            ..*self
        }
    }

    pub(crate) fn prev(&self) -> Self {
        Self {
            counter: (self.counter + self.frames_in_flight - 1) % self.frames_in_flight,
            ..*self
        }
    }

    pub fn idx(&self) -> usize {
        self.counter
    }
}

impl PartialEq for InFlightCounter {
    fn eq(&self, other: &Self) -> bool {
        self.frames_in_flight == other.frames_in_flight
        && self.counter == other.counter
    }
}

#[derive(Clone)]
pub(crate) struct InFlight<T: Sized>(pub(crate) Vec<T>);

impl<T: Sized> From<Vec<T>> for InFlight<T> {
    fn from(vec: Vec<T>) -> Self {
        Self(vec)
    }
}

impl<T: Sized> InFlight<T> {
    pub(crate) fn create<C: Fn(usize) -> T>(amount: usize, constructor: C) -> Self {
        let vec = (0..amount)
            .map(constructor)
            .collect();
        
        Self(vec)
    }
}

impl<T> Index<InFlightCounter> for InFlight<T> {
    type Output = T;

    fn index(&self, index: InFlightCounter) -> &Self::Output {
        assert_eq!(index.frames_in_flight, self.0.len());
        &self.0[index.counter]
    }
}

impl<T> IndexMut<InFlightCounter> for InFlight<T> {
    fn index_mut(&mut self, index: InFlightCounter) -> &mut Self::Output {
        assert_eq!(index.frames_in_flight, self.0.len());
        &mut self.0[index.counter]
    }
}
