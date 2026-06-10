use ndarray::{s, arr2, Array2}; 
use polars::prelude::*;
use polars::prelude::ParquetReader;
use dendritic::optimizer::prelude::*;
use dendritic::optimizer::regression::logistic::*; 
use dendritic::preprocessing::prelude::*; 

/*
fn print_type_of<T>(_: &T) {
    println!("Type: {}", std::any::type_name::<T>());
} */ 


// Testing & benchmarking model for multi class classification
pub struct IrisFlowersModel {

    /// Name of model
    name: String,

    /// Path of where dataset lives
    file_path: String,

    /// Training dataset as ndarray
    x: Array2<f64>,

    /// Target values as ndaarray
    y: Array2<f64>,

    /// Model associated with pipeline
    model: Logistic

}

impl ModelPipeline for IrisFlowersModel {
 
    fn register(name: &str) -> Self {
        IrisFlowersModel {
            name: name.to_string(),
            file_path: "data/iris.parquet".into(),
            x: Array2::zeros((150, 4)),
            y: Array2::zeros((150, 1)),
            model: Logistic::new(
                &Array2::zeros((150, 4)),
                &Array2::zeros((150, 1)),
                true,
                0.01
            ).unwrap()
        }
    }

}

impl Load for IrisFlowersModel {

    fn load(&mut self) {

        println!("Running load step for: {:?}", self.name); 

        let mut file = std::fs::File::open(&self.file_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();

        let df_select = df.select([
            "sepal_length_cm", 
            "sepal_width_cm", 
            "petal_length_cm", 
            "petal_width_cm"
        ]).unwrap();

        let df_target = df.select(["species_code"]).unwrap();
 
        self.x = df_select.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

        self.y = df_target.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

    }

}


impl Transform for IrisFlowersModel {

    fn transform(&mut self) {

        println!("Running transform step for: {:?}", self.name);

        let y_original = self.y.clone();
        let mut encoder = OneHot::new();
        let y_one_hot = encoder.transform(&y_original.view());
        self.y = y_one_hot; 

    }

}


impl Train for IrisFlowersModel {

    fn train(&mut self) {

        println!("Running train step for: {:?}", self.name); 

        self.model = Logistic::new(&self.x, &self.y, true, 0.001).unwrap(); 
        self.model.train_batch(3, 10, 1000);
        self.model.save("iris_classification").unwrap(); 
        println!("Model loss after training: {:?}", self.model.loss());
    }

}


impl Inference for IrisFlowersModel {

    fn inference(&mut self) {

        println!("Running inference step for: {:?}", self.name);
        println!("{:?}", self.y);
        let all_predictions = self.model.predict(&self.x); 
        println!("All predictions: {:?}", all_predictions);

    }

}
