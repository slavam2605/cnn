use cnn::NetworkLayer;

pub struct StartLayer {
    outputs: Vec<f64>,
    errors: Vec<f64>,
    block_size: usize
}

impl StartLayer {
    pub fn new(values: &[f64]) -> StartLayer {
        StartLayer {
            outputs: values.to_vec(),
            errors: values.to_vec(),
            block_size: 1
        }
    }

    pub fn new_block(values: &[f64], block_size: usize) -> StartLayer {
        StartLayer {
            outputs: values.to_vec(),
            errors: values.to_vec(),
            block_size: block_size
        }
    }
}

impl NetworkLayer for StartLayer {
    fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    fn block_size(&self) -> usize { 
        self.block_size
    }
    
    fn outputs(&self) -> &[f64] {
        &self.outputs
    }

    fn errors(&mut self) -> &mut [f64] {
        &mut self.errors
    }

    fn feed_forward(&mut self) {}

    fn train_backward(&mut self) {}

    fn set_data(&mut self, data: &[f64]) {
        self.outputs = data.to_vec();
    }
}

