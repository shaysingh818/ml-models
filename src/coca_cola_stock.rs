use ndarray::{s, Array2}; 
use polars::prelude::*;
use polars::prelude::ParquetReader;
use dendritic::optimizer::prelude::*;
use dendritic::preprocessing::prelude::*; 


pub struct CocaColaStockModel {

    /// Name of model
    name: String,

    /// Path of where training dataset lives
    train_file_path: String,

    /// Path of where testing dataset lives
    test_file_path: String,

    /// X train data set as ndarray
    x_train_features: Array2<f64>,

    /// X train data set as ndarray
    y_train_target: Array2<f64>,

    /// SGD Model (first iteration)
    sgd_model: SGD
}


impl ModelPipeline for CocaColaStockModel {
 
    fn register(name: &str) -> Self {

        CocaColaStockModel {
            name: name.to_string(), 
            train_file_path: "data/coca_cola_train.parquet".to_string(),
            test_file_path: "data/coca_cola_test.parquet".to_string(),
            x_train_features: Array2::zeros((0, 0)),
            y_train_target: Array2::zeros((0, 0)),
            sgd_model: SGD::new(
                &Array2::zeros((0, 0)),
                &Array2::zeros((0, 0)),
                0.01
            ).unwrap()
        }

    }
}


impl Load for CocaColaStockModel {

    fn load(&mut self) {

        println!("Running load step for: {:?}", self.name);

        let mut file = std::fs::File::open(&self.train_file_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap(); 

        let x_train_frame = df.select([
            "DATE_EPOCH",
            "OPEN_PRICE",
            "HIGH_PRICE",
            "LOW_PRICE"
        ]).unwrap();

        let y_target_frame = df.select(["CLOSE_PRICE"]).unwrap();

        self.x_train_features = x_train_frame.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

        self.y_train_target = y_target_frame.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

        println!("X features shape: {:?}", self.x_train_features.shape());
        println!("Y target shape: {:?}", self.y_train_target.shape());

    }

}


impl Transform for CocaColaStockModel {

    fn transform(&mut self) {

        println!("Running transform step for: {:?}", self.name);

        let scaled_x = self.x_train_features.clone();
        let mut min_max = StandardScalar::new();
        let encoded = min_max.transform(&scaled_x.view());
        self.x_train_features = encoded; 

    }

}


impl Train for CocaColaStockModel {

    fn train(&mut self) {

        println!("Running train step for: {:?}", self.name); 
        self.sgd_model = SGD::new(&self.x_train_features, &self.y_train_target, 0.0001).unwrap(); 
        self.sgd_model.train_batch(10, 200, 1000);
        self.sgd_model.save("coca_cola_sgd").unwrap();

    }

}


impl Inference for CocaColaStockModel {

    fn inference(&mut self) {

        let mut file = std::fs::File::open(&self.test_file_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap(); 

        let x_test_frame = df.select([
            "DATE_EPOCH",
            "OPEN_PRICE",
            "HIGH_PRICE",
            "LOW_PRICE"
        ]).unwrap();

        let y_test_frame = df.select(["CLOSE_PRICE"]).unwrap();

        let x_test_features = x_test_frame.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

        let y_test_target = y_test_frame.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

        let sample_data = x_test_features.slice(s![0..5, 0..4]);
        let sample_target = y_test_target.slice(s![0..5, 0..1]);

        let mut model = SGD::load("coca_cola_sgd").unwrap();
        let predictions = model.predict(&sample_data.to_owned());
        
        println!("Predictions");
        println!("{:?}", predictions);

        println!("Actual");
        println!("{:?}", sample_target);

    }

}
