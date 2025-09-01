use ndarray::{s, Array2}; 
use polars::prelude::*;
use polars::prelude::ParquetReader;
use dendritic::optimizer::prelude::*;
use dendritic::optimizer::regression::logistic::*; 
use dendritic::preprocessing::prelude::*; 

fn print_type_of<T>(_: &T) {
    println!("Type: {}", std::any::type_name::<T>());
} 


// Testing & benchmarking model for multi class classification
pub struct BreastCancerModel {

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



impl ModelPipeline for BreastCancerModel {
 
    fn register(name: &str) -> Self {
        BreastCancerModel {
            name: name.to_string(),
            file_path: "data/breast_cancer.parquet".into(),
            x: Array2::zeros((0, 0)),
            y: Array2::zeros((0, 0)),
            model: Logistic::new(
                &Array2::zeros((0, 0)),
                &Array2::zeros((0, 0)),
                true,
                0.01
            ).unwrap()
        }
    }

}



impl Load for BreastCancerModel {

    fn load(&mut self) {

        println!("Running load step for: {:?}", self.name); 

        let mut file = std::fs::File::open(&self.file_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();

        let df_select = df.select([
            "radius_mean",
            "texture_mean",
            "smoothness_mean",
            "compactness_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "smoothness_se",
            "compactness_se",
            "symmetry_se",
            "fractal_dimensions_se"
        ]).unwrap();


        let df_target = df.select(["diagnosis_code"]).unwrap(); 

        self.x = df_select.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

        self.y = df_target.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

    }

}


impl Transform for BreastCancerModel {

    fn transform(&mut self) {

        let temp_x = self.x.clone();
        let mut scalar = MinMax::new();
        let encoded = scalar.transform(&temp_x.view());
        self.x = encoded;

    }

}


impl Train for BreastCancerModel {

    fn train(&mut self) {
        
        self.model = Logistic::new(
            &self.x, 
            &self.y, 
            false, 
            0.001
        ).unwrap();

        self.model.train_batch(4, 10, 1000);

    }

}


impl Inference for BreastCancerModel {

    fn inference(&mut self) {
        
        let x_test = self.x.slice(s![0..12, 0..12]);
        let y_test = self.y.slice(s![0..12, 0..1]); 

        let test_1_predict = self.model.predict(&x_test.to_owned());
        println!("FIRST SET OF PREDICTIONS"); 
        println!("{:?}", test_1_predict);
        println!("{:?}", y_test);

        let x_test_2 = self.x.slice(s![140..150, 0..12]);
        let y_test_2 = self.y.slice(s![140..150, 0..1]);

        let test_2_predict = self.model.predict(&x_test_2.to_owned());
        println!("SECOND SET OF PREDICTIONS"); 
        println!("{:?}", test_2_predict);
        println!("{:?}", y_test_2);

    }

}



