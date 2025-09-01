use ndarray::{s, Array, Array2}; 
use polars::prelude::*;
use polars::prelude::ParquetReader;
use dendritic::optimizer::model::*;
use dendritic::optimizer::train::*; 
use dendritic::optimizer::optimizers::*;
use dendritic::optimizer::regression::sgd::*;
use dendritic::preprocessing::processor::*; 

/*
fn print_type_of<T>(_: &T) {
    println!("Type: {}", std::any::type_name::<T>());
} */ 


// Testing & benchmarking model for multi class classification
pub struct StudentPerformance {

    /// Name of model
    name: String,

    /// Path of where dataset lives
    file_path: String,

    /// Training dataset as ndarray
    x: Array2<f64>,

    /// Target values as ndaarray
    y: Array2<f64>,

    /// Training dataset as ndarray
    x_encode: StandardScalar,

    /// Target values as ndaarray
    y_encode: StandardScalar,

    /// Training dataset split
    training_data: (Array2<f64>, Array2<f64>),

    /// Testing data
    testing_data: (Array2<f64>, Array2<f64>),

    /// Model associated with pipeline
    model: SGD

}

impl ModelPipeline for StudentPerformance {
 
    fn register(name: &str) -> Self {

        let temp_x: Array2<f64> = Array2::zeros((0, 0)); 
        let temp_y: Array2<f64> = Array2::zeros((0, 0)); 

        StudentPerformance {
            name: name.to_string(),
            file_path: "data/student_performance.parquet".into(),
            x: temp_x.clone(),
            y: temp_y.clone(),
            x_encode: StandardScalar::new(),
            y_encode: StandardScalar::new(),
            training_data: (temp_x.clone(), temp_y.clone()),
            testing_data: (temp_x.clone(), temp_y.clone()),
            model: SGD::new(
                &temp_x,
                &temp_y,
                0.01
            ).unwrap()
        }
    }

}


impl Load for StudentPerformance {

    fn load(&mut self) {

        println!("Running load step for: {:?}", self.name); 

        let mut file = std::fs::File::open(&self.file_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();

        let df_select = df.select([
            "Hours Studied",
            "Previous Scores",
            "Sleep Hours",
            "Sample Question Papers Practiced",
        ]).unwrap();


        let df_target = df.select(["Performance Index"]).unwrap(); 

        self.x = df_select.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

        self.y = df_target.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

    }

}


impl Transform for StudentPerformance {

    fn transform(&mut self) {

        println!("Running transform step for: {:?}", self.name);

        let num_rows = self.x.nrows(); 
        if num_rows != self.y.nrows() {
            panic!("Number of rows for sample features and target unequal");
        }

        let x_enc = self.x_encode.transform(&self.x.view());
        let y_enc = self.y_encode.transform(&self.y.view()); 
        
        self.x = x_enc;
        self.y = y_enc; 

        let train_split = 0.8 * num_rows as f64; 
        let test_split = 0.2 * num_rows as f64;

        self.training_data = (
            self.x.slice(s![0..train_split as usize, ..]).to_owned(), 
            self.y.slice(s![0..train_split as usize, ..]).to_owned()
        );

        self.testing_data = (
            self.x.slice(s![train_split as usize..num_rows, ..]).to_owned(), 
            self.y.slice(s![train_split as usize..num_rows, ..]).to_owned()
        );


    }

}


impl Train for StudentPerformance {

    fn train(&mut self) {

        println!("Running train step for: {:?}", self.name); 

        self.model = SGD::new(
            &self.training_data.0, 
            &self.training_data.1, 
            0.001
        ).unwrap();

        let mut opt = Adam::default(&self.model);

        self.model.train_batch_with_optimizer(10, 128, 1000, &mut opt);
        self.model.save("models/student_performance").unwrap();

    }

}


impl Inference for StudentPerformance {

    fn inference(&mut self) {

        println!("Running inference step for: {:?}", self.name); 

        let mut loaded = SGD::load("models/student_performance").unwrap();

        let x1 = self.testing_data.0.slice(s![0..5, ..]);
        let y1 = self.testing_data.1.slice(s![0..5, ..]);

        println!("First set of predictions");
        println!("{:?}", self.y_encode.inverse_transform(&y1.view()));
        let predicted = loaded.predict(&x1.to_owned());
        println!("{:?}", self.y_encode.inverse_transform(&predicted.view())); 

        let x2 = self.testing_data.0.slice(s![5..10, ..]);
        let y2 = self.testing_data.1.slice(s![5..10, ..]);

        println!("Second set of predictions");
        println!("{:?}", self.y_encode.inverse_transform(&y2.view()));
        let predicted = loaded.predict(&x2.to_owned());
        println!("{:?}", self.y_encode.inverse_transform(&predicted.view())); 


        let x3 = self.testing_data.0.slice(s![10..15, ..]);
        let y3 = self.testing_data.1.slice(s![10..15, ..]);

        println!("Third set of predictions");
        println!("{:?}", self.y_encode.inverse_transform(&y3.view()));
        let predicted = loaded.predict(&x3.to_owned());
        println!("{:?}", self.y_encode.inverse_transform(&predicted.view())); 


    }

}
