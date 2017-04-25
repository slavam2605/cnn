use std::cell::RefCell;
use std::rc::Rc;

use full_layer::FullLayer;
use start_layer::StartLayer;

pub trait NetworkLayer {
    fn num_outputs(&self) -> usize;
    fn block_size(&self) -> usize;
    fn outputs(&self) -> &[f64];
    fn errors(&mut self) -> &mut [f64];
    fn feed_forward(&mut self);
    fn train_backward(&mut self);
    fn set_data(&mut self, &[f64]);

    fn layers_count(&self) -> usize {
        self.num_outputs() / self.block_size() / self.block_size()
    }

    fn clear_errors(&mut self) {
        let n = self.num_outputs();
        let errors = self.errors();
        for i in 0..n {
            errors[i] = 0.0;
        }
    }
}

pub struct NeuralNetwork {
    layers: Vec<Rc<RefCell<Box<NetworkLayer>>>>
}

impl NeuralNetwork {
    pub fn new(num_inputs: usize) -> NeuralNetwork {
        let mut result = NeuralNetwork {
            layers: Vec::new()
        };
        result.layers.push(Rc::new(RefCell::new(box StartLayer::new(&vec![0.0; num_inputs]))));
        let layer: Rc<RefCell<Box<NetworkLayer>>> = Rc::new(RefCell::new(box FullLayer::new(2, &result.layers[0])));
        result.layers.push(layer);
        let layer: Rc<RefCell<Box<NetworkLayer>>> = Rc::new(RefCell::new(box FullLayer::new(2, &result.layers[1])));
        result.layers.push(layer);
        let layer: Rc<RefCell<Box<NetworkLayer>>> = Rc::new(RefCell::new(box FullLayer::new(2, &result.layers[2])));
        result.layers.push(layer);
        result
    }

    pub fn train(&mut self, data: &[f64], ans: &[f64]) {
        self.layers[0].borrow_mut().set_data(ans);
        for layer in &*self.layers {
            layer.borrow_mut().feed_forward();
        }
        let last_layer = &self.layers[self.layers.len() - 1];
        for i in 0..ans.len() {
            let mut last_layer = last_layer.borrow_mut();
            last_layer.errors()[i] = last_layer.outputs()[i] - ans[i];
        }
        for layer in self.layers.iter().rev() {
            layer.borrow_mut().train_backward();
        }
    }

    pub fn predict(&mut self, data: &[f64]) -> Box<[f64]> {
        self.layers[0].borrow_mut().set_data(data);
        for layer in &*self.layers {
            layer.borrow_mut().feed_forward();
        }
        box (*self.layers[self.layers.len() - 1].borrow().outputs())
    }
} 