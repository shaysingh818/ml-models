use ndarray::{arr2, Array, Array2, Axis}; 
use polars::prelude::*;
use polars::prelude::ParquetReader;
use dendritic::optimizer::model::*;
use dendritic::optimizer::train::*;  
use dendritic::optimizer::regression::logistic::*;
use dendritic::preprocessing::processor::*; 

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

        let mut x: Array2<f64> = Array2::zeros((150, 4));
        let mut y: Array2<f64> = Array2::zeros((150, 1));

        let mut file = std::fs::File::open(&self.file_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();

        let cols = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"];
        for (i, col_name) in cols.iter().enumerate() {

            let col_df = df.column(col_name).unwrap().f64().unwrap();
            let col_df_vec: Vec<f64> = col_df.into_no_null_iter().collect();

            let col = Array::from_vec(col_df_vec);
            x.index_axis_mut(Axis(1), i).assign(&col);
        }

        let y_col_df = df.column("species_code").unwrap().f64().unwrap();
        let y_col_df_vec: Vec<f64> = y_col_df.into_no_null_iter().collect();

        let col = Array::from_vec(y_col_df_vec);
        y.index_axis_mut(Axis(1), 0).assign(&col);

        self.x = x; 
        self.y = y; 

    }

}


impl Transform for IrisFlowersModel {

    fn transform(&mut self) {

        println!("Running transform step for: {:?}", self.name);

        let y_original = self.y.clone();
        let mut encoder = OneHotEncoding::new(&y_original).unwrap();
        let y_one_hot = encoder.encode();
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

        let class_0 = arr2(&[
            [5.0, 3.5, 1.4, 0.2],
            [4.3, 3.0, 1.1, 0.1],
            [4.6, 3.4, 1.4, 0.3]
        ]); 

        let class_0_predict = self.model.predict(&class_0); 
        println!("First class predictions");
        println!("{:?}", class_0_predict);

        let class_1 = arr2(&[
            [6.4, 2.9, 4.3, 1.3],
            [6.8, 2.8, 4.8, 1.4],
            [6.7, 3.0, 5.0, 1.7]
        ]); 

        let class_1_predict = self.model.predict(&class_1); 
        println!("Second class predictions");
        println!("{:?}", class_1_predict);

        let class_2 = arr2(&[
            [6.4, 2.7, 5.3, 1.9],
            [7.7, 3.8, 6.7, 2.2],
            [7.9, 3.8, 6.4, 2.0]
        ]); 

        let class_2_predict = self.model.predict(&class_2); 
        println!("Third class predictions");
        println!("{:?}", class_2_predict); 


    }

}
