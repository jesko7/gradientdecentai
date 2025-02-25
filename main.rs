use Random::{self, Rng};
use mnist::MnistBuilder;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::time::Instant;

fn index_of_max_value(list: Vec<f64>) -> i32 {
    let mut index = 0;

    for (i, item) in list.iter().enumerate() {
        if item > &list[index] {
            index = i;
        }
    }

    index as i32
}


#[derive(Debug, Clone, Serialize, Deserialize)]
enum ActivationFunction {
    Sigmoid,
    Relu,
    Step,
    HyperbolicTangent,
    SiLU
}

impl ActivationFunction {
    fn eval(&self, weighted_input: f64) -> f64 {
        match self {
            ActivationFunction::Relu => {
                0f64.max(weighted_input)
            },
            ActivationFunction::Sigmoid => {
                1. / (1. + (-weighted_input).exp())
            }
            ActivationFunction::Step => {
                if weighted_input > 0. {
                    1.
                }
                else {
                    0.
                }
            }
            ActivationFunction::HyperbolicTangent => {
                let e2w = (2. * weighted_input).exp();
                (e2w - 1.) / (e2w + 1.)
            }
            ActivationFunction::SiLU => {
                weighted_input / (1. + (-weighted_input).exp())
            }
        }
    }

    fn derivative(&self, weighted_input: f64) -> f64 {
        match self {
            ActivationFunction::Relu => {
                0f64.max(weighted_input)
            },
            ActivationFunction::Sigmoid => {
                let sigmoid_x = self.eval(weighted_input);
                sigmoid_x * (1.0 - sigmoid_x)
            }
            ActivationFunction::Step => {
                0.
            }
            ActivationFunction::HyperbolicTangent => {
                let e2w = (2. * weighted_input).exp();
                (e2w - 1.) / (e2w + 1.)
            }
            ActivationFunction::SiLU => {
                weighted_input / (1. + (-weighted_input).exp())
            }
        }
    }
}



#[derive(Debug, Clone, Serialize, Deserialize)]
struct DataPoint {
    inputs: Vec<f64>,
    expected_outputs: Vec<f64>
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct LayerLearnData {
    inputs: Vec<f64>,
    weighted_inputs:Vec<f64>,
    activations: Vec<f64>,
    node_values: Vec<f64>
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct NeuralNetworkLearnData {
    layerdata: Vec<LayerLearnData>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Layer {
    incoming_nodes: i32,
    outgoing_nodes: i32,

    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,

    cost_gradient_w: Vec<Vec<f64>>,
    cost_gradient_b: Vec<f64>,

    activation_function: ActivationFunction
}

impl Layer {
    fn new(incoming_nodes: i32, outgoing_nodes: i32, activation_function: ActivationFunction) -> Layer {
        let mut rng = Random::prelude::thread_rng();
        
        let mut layer = Layer { incoming_nodes, outgoing_nodes, weights: vec![], biases: vec![], activation_function, cost_gradient_w: vec![], cost_gradient_b: vec![] };
        
        for _ in 0..outgoing_nodes {
            let mut current_wheights = vec![];
            for _ in 0.. incoming_nodes{
                let val: f64 = rng.gen_range(-100000..100000) as f64 / 100000.;
                current_wheights.push(val / (incoming_nodes as f64).sqrt());
            }
            layer.weights.push(current_wheights);
        }

        for _ in 0..outgoing_nodes {
            let val: f64 = rng.gen_range(-100000..100000) as f64 / 100000.;
            layer.biases.push(val / (incoming_nodes as f64).sqrt());
        }

        layer
    }



    fn calculate_outputs(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut activations = vec![];

        for node_out in 0..self.outgoing_nodes {
            let mut weighted_input = self.biases[node_out as usize];

            for node_in in 0..self.incoming_nodes {
                weighted_input += inputs[node_in as usize] * self.weights[node_out as usize][node_in as usize];
            }

            activations.push(self.activation_function.eval(weighted_input));
        }

        activations
    }

    fn calculate_outputs_layer_learn(&self, inputs: Vec<f64>, learn_data: &mut LayerLearnData) -> Vec<f64> {
        learn_data.inputs = inputs.clone();

        for node_out in 0..self.outgoing_nodes {
            let mut weighted_input = self.biases[node_out as usize];
            for node_in in 0..self.incoming_nodes {
                weighted_input += inputs[node_in as usize] * self.weights[node_out as usize][node_in as usize];
            }
            learn_data.weighted_inputs[node_out as usize] = weighted_input;
        }

        for (i, weighted_input) in learn_data.weighted_inputs.iter().enumerate() {
            learn_data.activations[i] = self.activation_function.eval(*weighted_input);
        }

        learn_data.activations.clone()
    }

    fn calculate_output_layer_node_values(&self, learn_data: &mut LayerLearnData, expected_activations: &Vec<f64>) {
        let mut node_values = vec![];

        for i in 0..learn_data.weighted_inputs.len() {
            let cost_derivative = Layer::node_cost_derivative(learn_data.activations[i], expected_activations[i]);
            let activation_derivative = self.activation_function.derivative(learn_data.weighted_inputs[i]);
            node_values.push(cost_derivative * activation_derivative);
        }

        learn_data.node_values = node_values;
    }

    fn calculate_hidden_layer_node_values(&self, learn_data: &mut LayerLearnData, old_layer: &Layer, old_node_values: &Vec<f64>) {
        let mut node_values = vec![];

        for new_node_index in 0..self.outgoing_nodes {
            let mut new_node_value = 0.;
            for old_node_index in 0..old_node_values.len() {
                let weighted_input_derivative = old_layer.weights[old_node_index][new_node_index as usize];
                new_node_value += weighted_input_derivative * old_node_values[old_node_index];
            }
            new_node_value *= self.activation_function.derivative(learn_data.weighted_inputs[new_node_index as usize]);
            node_values.push(new_node_value);
        }

        learn_data.node_values = node_values;
    }


    fn node_cost(&self, output_activation: f64, expected_activation: f64) -> f64 {
        let error = output_activation - expected_activation;
        error * error
    } 

    fn node_cost_derivative(output_activation: f64, expected_activation: f64) -> f64 {
        return 2. * (output_activation - expected_activation);
    }

    fn apply_gradients(&mut self, learnrate: f64) {


        for node_out in 0..self.outgoing_nodes {
            self.biases[node_out as usize] -= self.cost_gradient_b[node_out as usize] * learnrate;
            for node_in in 0.. self.incoming_nodes {
                self.weights[node_out as usize][node_in as usize] -= self.cost_gradient_w[node_out as usize][node_in as usize] * learnrate;
            }
        }
    }

    fn update_gradients(&mut self, learn_data: &LayerLearnData) {
        let mut gradient = vec![];
        
        for node_out in 0..self.outgoing_nodes {
            gradient.push(vec![]);
            let node_value = learn_data.node_values[node_out as usize];
            for node_in in 0..self.incoming_nodes {
                let derivative_cost_wrt_weight = learn_data.inputs[node_in as usize] * node_value;
                gradient[node_out as usize].push(derivative_cost_wrt_weight);
            }
        }

        self.cost_gradient_w = gradient;

        let mut gradient = vec![];
        
        for node_out in 0..self.outgoing_nodes {
            let derivative_cost_wrt_bias = 1. * learn_data.node_values[node_out as usize];
            gradient.push(derivative_cost_wrt_bias);
        }

        self.cost_gradient_b = gradient;


        
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct NeuralNetwork {
    layers: Vec<Layer>,
    activation_function: ActivationFunction,
    network_learn_data: NeuralNetworkLearnData
}

impl NeuralNetwork {
    fn new(layer_sizes: Vec<i32>, activation_function: ActivationFunction) -> NeuralNetwork {
        let mut layers = vec![];

        let mut network_learn_data = NeuralNetworkLearnData { layerdata: vec![] };

        for size in layer_sizes.windows(2) {
            layers.push(Layer::new(size[0], size[1], activation_function.clone()));
            network_learn_data.layerdata.push(LayerLearnData { inputs: vec![0.].repeat(size[0] as usize), weighted_inputs: vec![0.].repeat(size[1] as usize), activations: vec![0.].repeat(size[1] as usize), node_values: vec![0.].repeat(size[1] as usize) });
        }
        
        NeuralNetwork {
            layers,
            activation_function,
            network_learn_data
        }
    }

    fn calculate_outputs(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut inputs = inputs;
        
        for layer in self.layers.iter() {
            inputs = layer.calculate_outputs(inputs);
        }

        inputs
    }

    fn classify(&self, inputs: Vec<f64>) -> i32 {
        let outputs = self.calculate_outputs(inputs);
        index_of_max_value(outputs)
    }

    fn cost(&self, data_point: &DataPoint) -> f64 {
        let outputs = self.calculate_outputs(data_point.inputs.clone());
        let output_layer = self.layers.last().unwrap();

        let mut cost = 0.;

        for (i, node) in outputs.iter().enumerate() {
            cost += output_layer.node_cost(*node, data_point.expected_outputs[i])
        }

        cost
    }

    fn total_cost(&self, data_points: Vec<DataPoint>) -> f64 {
        let mut total_cost = 0.;

        for data_point in data_points.iter() {
            total_cost += self.cost(data_point);
        }

        return total_cost / data_points.len() as f64;
    }


    /*
    fn Learn(&mut self, data_points: Vec<DataPoint>, learnrate: f64) {
        let origional_cost = self.total_cost(data_points.clone());

        for layer in 0..self.layers.len() {
            for nodeIn in 0..self.layers[layer].incoming_nodes as usize {
                for nodeOut in 0..self.layers[layer].outgoing_nodes as usize {
                    self.layers[layer].weights[nodeIn][nodeOut] += H;
                    let delta_cost = self.total_cost(data_points.clone()) - origional_cost;
                    self.layers[layer].weights[nodeIn][nodeOut] -= H;
                    self.layers[layer].cost_gradient_w[nodeIn][nodeOut] = delta_cost / H;
                }
            }


            for biasIndex in 0..self.layers[layer].biases.len() {
                self.layers[layer].biases[biasIndex] += H;
                let delta_cost = self.total_cost(data_points.clone()) - origional_cost;
                self.layers[layer].biases[biasIndex] -= H;
                self.layers[layer].cost_gradient_b[biasIndex] = delta_cost / H;
            } 
        }

        for layer in self.layers.iter_mut() {
            layer.apply_gradients(learnrate);
        }
    }
     */
    fn train(&mut self, data_points: Vec<DataPoint>, learnrate: f64, epochs: usize, print_epochs: i32, divide_learningrate: f64) {
        let mut learnrate = learnrate;
        
        for done in 0..epochs {
            let start = Instant::now();

            self.learn(&data_points, learnrate);

            learnrate /= divide_learningrate;
            if done as i32 % print_epochs == 0 {
                let time = start.elapsed().as_secs();
                let test_data = &data_points[0..10];
                let total_seconds = (epochs - done) * time as usize;

                let hours = total_seconds / 3600;
                let minutes = (total_seconds % 3600) / 60;
                let seconds = total_seconds % 60;

                println!("epoch: {}/{epochs} --------- \n     cost: {}\n     learning rate: {}\n     time per epoch: {time}s\n     eta: {seconds}s {minutes}m {hours}h", done, self.total_cost(test_data.to_vec()), learnrate, );
            
                print!("[");

                let percent = done / epochs * 100;

                for _ in 0..percent {
                    print!("#");
                }
                for _ in 0..100 - percent {
                    print!(" ");
                }

                print!("]   {}%\n ", percent);

                let _ = std::io::stdout().flush();       
            }
        }
        
    }


    fn learn(&mut self, data_points: &Vec<DataPoint>, learnrate: f64) { 
        for data_point in data_points.iter() { 
            self.update_gradients(data_point);
            

            for layer in self.layers.iter_mut() {
                layer.apply_gradients(learnrate);
            }
        } 
    }


    fn update_gradients(&mut self, data_point: &DataPoint) {

        
        let mut inputs_to_next_layer = data_point.inputs.clone();

        for i in 0..self.layers.len() {
            inputs_to_next_layer = self.layers[i].calculate_outputs_layer_learn(inputs_to_next_layer, &mut self.network_learn_data.layerdata[i])
        }

        let output_layer_index = self.layers.len() - 1;
        let output_layer = &mut self.layers[output_layer_index];
        let mut output_layer_learn_data = &mut self.network_learn_data.layerdata[output_layer_index];

        
        output_layer.calculate_output_layer_node_values(&mut output_layer_learn_data, &data_point.expected_outputs);
        output_layer.update_gradients(&mut output_layer_learn_data); // you can put this into a thread

        for i in 1..output_layer_index + 1 {
            let i = output_layer_index - i;

            let mut layer_learn_data = self.network_learn_data.layerdata[i].clone();
            let old_node_values =  &self.network_learn_data.layerdata[i + 1].node_values;

            self.layers[i].calculate_hidden_layer_node_values(&mut layer_learn_data, &self.layers[i + 1], old_node_values);
            self.layers[i].update_gradients(&layer_learn_data); // you can put this into a thread
            self.network_learn_data.layerdata[i] = layer_learn_data;
        }


    }


    fn save_to_file(&self, filename: &str) -> io::Result<()> {
        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }

    fn load_from_file(filename: &str) -> io::Result<NeuralNetwork> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let neural_network = serde_json::from_reader(reader)?;
        Ok(neural_network)
    }
}


fn tupel_to_data_point(x: Vec<(Vec<f64>, Vec<f64>)>) -> Vec<DataPoint> {
    let mut data_points = vec![];

    for (inputs, expected_outputs) in x {
        data_points.push(DataPoint {inputs, expected_outputs});
    }

    data_points
}

fn round_to_decimal_places(value: f64, decimal_places: u32) -> f64 {
    let factor = 10_f64.powi(decimal_places as i32);
    (value * factor).round() / factor
}
use image::{GrayImage, Luma};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use image::imageops;


fn augment_mnist_image(image: &GrayImage, label: u8) -> GrayImage {
    let mut rng = Random::thread_rng();


    let scale_factor = rng.gen_range(0.95..=1.05);


    let mut rotation_angle = rng.gen_range(-5.0..=5.0);

    if label == 1 {
        rotation_angle = 0.;
    }


    let (width, height) = (image.width() as f32, image.height() as f32);
    let scaled_width = (width * scale_factor).round() as u32;
    let scaled_height = (height * scale_factor).round() as u32;

    let scaled_image = imageops::resize(
        image,
        scaled_width,
        scaled_height,
        imageops::FilterType::Nearest,
    );


    let mut centered_image = GrayImage::new(28, 28);
    let x_offset = (28 - scaled_width.min(28)) / 2;
    let y_offset = (28 - scaled_height.min(28)) / 2;

    for y in 0..scaled_height.min(28) {
        for x in 0..scaled_width.min(28) {
            centered_image.put_pixel(
                x + x_offset,
                y + y_offset,
            {

                    let mut pixel = *scaled_image.get_pixel(x, y);

                    if pixel.0[0] > 200 {
                        pixel = Luma([255u8])
                    }
                    pixel
                },
            );
        }
    }


    let rotated_image = rotate_about_center(
        &centered_image,
        (rotation_angle as f32).to_radians(),
        Interpolation::Bilinear,
        Luma([0]),
    );


    let mut noisy_image = rotated_image.clone();
    for y in 0..28 {
        for x in 0..28 {
            let pixel = noisy_image.get_pixel(x, y);
            let intensity = if pixel[0] == 0 {
                rng.gen_range(0..23)
            } else {
                pixel[0]
            };
            noisy_image.put_pixel(x, y, Luma([intensity]));
        }
    }

    noisy_image
}


fn augment_mnist_dataset(images: &[u8], labels: &[u8]) -> Vec<GrayImage> {
    images
        .chunks(28 * 28).enumerate()
        .map(|(i, chunk)| {
            let image = GrayImage::from_raw(28, 28, chunk.to_vec())
                .expect("Chunk size must match image dimensions.");
            augment_mnist_image(&image, labels[i])
        })
        .collect()
}






fn main() {
    let mnist = MnistBuilder::new()
        .base_path("data/")
        .training_images_filename("train-images.idx3-ubyte")
        .training_labels_filename("train-labels.idx1-ubyte")
        .test_images_filename("t10k-images.idx3-ubyte")
        .test_labels_filename("t10k-labels.idx1-ubyte")
        .label_format_digit()
        .training_set_length(60000)
        .test_set_length(10_000)
        .finalize();


    let trn_lbl = mnist.trn_lbl;
    let trn_img = augment_mnist_dataset(&mnist.trn_img, &trn_lbl);
    let tst_lbl = mnist.tst_lbl;
    let tst_img = augment_mnist_dataset(&mnist.tst_img, &tst_lbl);
   

    let mut training_data: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut testing_data: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();

    for (i, &label) in trn_lbl.iter().enumerate() {
        let image = trn_img[i].as_flat_samples().samples.iter().map(|x| *x as f64 / 256.).collect::<Vec<f64>>();
        

        let mut label_vec = vec![0.0; 10];
        label_vec[label as usize] = 1.0;
        training_data.push((image, label_vec));
    }

    for (i, &label) in tst_lbl.iter().enumerate() {
        let image = tst_img[i].as_flat_samples().samples.iter().map(|x| *x as f64 / 256.).collect::<Vec<f64>>();


        let mut label_vec = vec![0.0; 10];
        label_vec[label as usize] = 1.0;
        testing_data.push((image, label_vec));
    }

    //training_data.append(&mut testing_data.clone());

    println!("trn: {}, tst: {}", training_data.len(), testing_data.len());

    let mut neural_network = NeuralNetwork::new(vec![784, 200, 10], ActivationFunction::Relu);

    //let mut neural_network = NeuralNetwork::load_from_file("NeuralNetwork").unwrap();
    let training_data = tupel_to_data_point(training_data);
    let testing_data = tupel_to_data_point(testing_data);

    neural_network.train(training_data, 0.5, 20, 1, 1.1);

    
    println!("test cost: {}", neural_network.total_cost(testing_data.clone()));

    let _ = neural_network.save_to_file("NeuralNetwork");
}
