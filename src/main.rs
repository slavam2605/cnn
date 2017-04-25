#![feature(associated_type_defaults)]
#![feature(box_syntax)]
#![feature(conservative_impl_trait)]

extern crate rand;

mod cnn;
mod start_layer;
mod full_layer;

use cnn::NeuralNetwork;
use rand::Rng;

fn main() {
    let mut data = [0.0; 3];
    let mut network = NeuralNetwork::new(3);
     
    let mut rng = rand::thread_rng();
    for i in 0..100000 {
        let a = rng.gen::<f64>();
        let b = rng.gen::<f64>();
        let c = rng.gen::<f64>();
        let d = (a + b) % 2.0 * c;
        let e = 1.0 - (1.0 - a) * (1.0 - b) * (1.0 - c);
        data[0] = a * 2.0 - 1.0;
        data[1] = b * 2.0 - 1.0;
        data[2] = c * 2.0 - 1.0;
        network.train(&data, &[d * 2.0 - 1.0, e * 2.0 - 1.0]);
    }
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                println!("{}, {}, {}: {:?}", i, j, k, network.predict(&[
                    i as f64 * 2.0 - 1.0,
                    j as f64 * 2.0 - 1.0,
                    k as f64 * 2.0 - 1.0
                ]));
            }
        }
    }
    
    println!("Hello, world!");
}
