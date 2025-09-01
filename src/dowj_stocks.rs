use ndarray::{s, Array2}; 
use polars::prelude::*;
use polars::prelude::ParquetReader;
use dendritic::optimizer::prelude::*; 
use dendritic::preprocessing::prelude::*; 

/*
fn print_type_of<T>(_: &T) {
    println!("Type: {}", std::any::type_name::<T>());
} */ 


// Testing & benchmarking model for multi class classification
pub struct DOWJModel {

    /// Name of model
    name: String,

    /// Path of where dataset lives
    file_path: String,

    /// Training dataset as ndarray
    x: Array2<f64>,

    /// Target values as ndaarray
    y: Array2<f64>,

    /// Model associated with pipeline
    model: SGD

}

impl ModelPipeline for DOWJModel {
 
    fn register(name: &str) -> Self {
        DOWJModel {
            name: name.to_string(),
            file_path: "data/dowj_stocks.parquet".into(),
            x: Array2::zeros((0, 0)),
            y: Array2::zeros((0, 0)),
            model: SGD::new(
                &Array2::zeros((0, 0)),
                &Array2::zeros((0, 0)),
                0.01
            ).unwrap()
        }
    }

}


impl Load for DOWJModel {

    fn load(&mut self) {

        println!("Running load step for: {:?}", self.name); 

        let mut file = std::fs::File::open(&self.file_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();

        let df_select = df.select([
            "Open",
            "High",
            "Low"
        ]).unwrap();


        let df_target = df.select(["Close"]).unwrap(); 

        self.x = df_select.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

        self.y = df_target.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

    }

}


impl Transform for DOWJModel {

    fn transform(&mut self) {

        println!("Running transform step for: {:?}", self.name); 

        let temp_x = self.x.clone();
        let mut scalar = MinMax::new();
        let encoded = scalar.transform(&temp_x.view());

        let temp_y = self.y.clone();
        let mut scalar_target = MinMax::new();
        let encoded_2 = scalar_target.transform(&temp_y.view());

        self.x = encoded; 
        self.y = encoded_2; 

    }

}


impl Train for DOWJModel {

    fn train(&mut self) {

        println!("Running train step for: {:?}", self.name); 

        self.model = SGD::new(&self.x, &self.y, 0.8).unwrap();
        self.model.train_batch(20, 10000, 100);
        self.model.save("dowj_stocks").unwrap();

    }

}


impl Inference for DOWJModel {

    fn inference(&mut self) {

        println!("Running train step for: {:?}", self.name); 

        let mut loaded = SGD::load("models/dowj_stocks").unwrap();
        let sample_data = self.x.slice(s![0..5, 0..3]);
        let sample_target = self.y.slice(s![0..5, 0..1]);

        println!("{:?}", sample_target);
        println!("{:?}", loaded.predict(&sample_data.to_owned())); 

    }

}
