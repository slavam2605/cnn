extern crate rand;

use cnn::NetworkLayer;
use self::rand::distributions::{IndependentSample, Range};
use std::cell::RefCell;
use std::rc::Rc;

pub struct FullLayer {
    prev: Rc<RefCell<Box<NetworkLayer>>>,
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    raw_outputs: Vec<f64>,
    outputs: Vec<f64>,
    errors: Vec<f64>,
    n: usize
}

impl FullLayer {
    pub fn new(n: usize, prev: &Rc<RefCell<Box<NetworkLayer>>>) -> FullLayer {
        let m = prev.borrow().num_outputs();
        let mut layer = FullLayer {
            prev: prev.clone(),
            weights: vec![vec![0.0; n]; m],
            bias: vec![0.0; n],
            outputs: vec![0.0; n],
            raw_outputs: vec![0.0; n],
            errors: vec![0.0; n],
            n: n,
        };
        let range = Range::new(-1.0 / (m as f64).sqrt(), 1.0 / (m as f64).sqrt());
        let mut rng = rand::thread_rng();
        for i in 0..m {
            for j in 0..n {
                layer.weights[i][j] = range.ind_sample(&mut rng);
            }
        }
        layer
    }
}

const SIGMOID_A: f64 = 2.1;
const LEARN_RATE: f64 = 0.01;

fn f(x: f64) -> f64 {
    2.0 / (1.0 + (-SIGMOID_A * x).exp()) - 1.0
}

fn df(x: f64) -> f64 {
    let eax = (SIGMOID_A * x).exp();
    2.0 * SIGMOID_A * eax / (eax + 1.0) / (eax + 1.0)
}

impl NetworkLayer for FullLayer {
    fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    fn block_size(&self) -> usize { 1 }
    
    fn outputs(&self) -> &[f64] {
        &self.outputs
    }

    fn errors(&mut self) -> &mut [f64] {
        &mut self.errors
    }

    fn feed_forward(&mut self) {
        let m = self.prev.borrow().num_outputs();
        let n = self.n;
        for j in 0..n {
            self.raw_outputs[j] = self.bias[j];
        }
        for i in 0..m {
            for j in 0..n {
                self.raw_outputs[j] += self.weights[i][j] * self.prev.borrow().outputs()[i]; 
            }
        }
        for j in 0..n {
            self.outputs[j] = f(self.raw_outputs[j]);
        }
    }

    fn train_backward(&mut self) {
        let prev = self.prev.borrow_mut();
        prev.clear_errors();
        let n = self.n;
        let m = prev.num_outputs();
        for i in 0..n {
            for j in 0..m {
                prev.errors()[j] += self.weights[j][i] * self.errors[i]; 
            }
        }
        for i in 0..n {
            for j in 0..m {
                self.weights[j][i] -= LEARN_RATE * self.errors[i] * df(self.raw_outputs[i]) * prev.outputs()[j]; 
            }
        }
        for i in 0..n {
            self.bias[i] -= LEARN_RATE * self.errors[i] * df(self.raw_outputs[i]);
        }
        prev.train_backward();
    }

    fn set_data(&mut self, data: &[f64]) {}
}