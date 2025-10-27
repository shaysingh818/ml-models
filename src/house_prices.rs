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
    test_file_path: String
}


impl ModelPipeline for CocaColaStockModel {
 
    fn register(name: &str) -> Self {

        CocaColaStockModel {
            name: name.to_string(), 
            train_file_path: "data/coca_cola_train.parquet".to_string(),
            test_file_path: "data/coca_cola_test.parquet".to_string(),
        }

    }
}


impl Load for CocaColaStockModel {

    fn load(&mut self) {

        println!("Running load step for: {:?}", self.name); 

    }

}


impl Transform for CocaColaStockModel {

    fn transform(&mut self) {

        println!("Running transform step for: {:?}", self.name);

    }

}


impl Train for CocaColaStockModel {

    fn train(&mut self) {

        println!("Running train step for: {:?}", self.name); 

    }

}


impl Inference for CocaColaStockModel {

    fn inference(&mut self) {

        println!("Running inference step for: {:?}", self.name); 

    }

}
