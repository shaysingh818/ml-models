use ndarray::{s, Array2, Array1, Axis};
use rand::seq::SliceRandom;
use polars::prelude::*;
use polars::prelude::ParquetReader;
use dendritic::preprocessing::prelude::*;
use dendritic::optimizer::prelude::*;
use dendritic::optimizer::regression::logistic::*; 


pub struct TitanicModel {

    /// Path of where training dataset lives
    dataset: String,

    /// Training dataset as ndarray
    x_train: Array2<f64>,

    /// Target values as ndaarray
    y_train: Array2<f64>,

    /// Training dataset as ndarray
    x_test: Array2<f64>,

    /// Target values as ndaarray
    y_test: Array2<f64>,

    /// Model type
    model: Logistic,

}

impl TitanicModel {

    pub fn new() -> Self {
        TitanicModel {
            dataset: "data/titanic.parquet".to_string(),
            x_train: Array2::zeros((0, 0)),
            y_train: Array2::zeros((0, 0)),
            x_test: Array2::zeros((0, 0)),
            y_test: Array2::zeros((0, 0)),
            model: Logistic::new(
                &Array2::zeros((0, 0)),
                &Array2::zeros((0, 0)),
                false,
                0.00001
            ).unwrap(),
        }
    }

    pub fn load_data(&mut self, test_size: f64) {

        let mut file = std::fs::File::open(&self.dataset).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();

        let df_select = df.select([
            "SEX",
            "AGE_NORM",
            "SIBLING_SPOUSE_ABOARD",
            "PARCH",
            "FARE",
            "PCLASS_1",
            "PCLASS_2",
            "EMBARKED_S",
            "EMBARKED_C",
            "EMBARKED_Q"
        ]).unwrap();

        let df_target = df.select(["SURVIVED"]).unwrap(); 

        let x = df_select.
            to_ndarray::<Float64Type>(IndexOrder::C).unwrap();

        let y = df_target.
            to_ndarray::<Float64Type>(IndexOrder::C).unwrap();


        let n_samples = x.nrows();
        let n_train = n_samples - (n_samples as f64 * test_size).round() as usize;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rand::rng());

        let x_shuffled = x.select(Axis(0), &indices);
        let y_shuffled = y.select(Axis(0), &indices);

        self.x_train = x_shuffled.slice(s![..n_train, ..]).to_owned();
        self.x_test = x_shuffled.slice(s![n_train.., ..]).to_owned();
 
        self.y_train = y_shuffled.slice(s![..n_train, ..]).to_owned();
        self.y_test = y_shuffled.slice(s![n_train.., ..]).to_owned();

        println!("Shapes: ");
        println!("X Shapes: {:?}, {:?}", self.x_train.shape(), self.x_test.shape());
        println!("Y Shapes: {:?}, {:?}", self.y_train.shape(), self.y_test.shape());
    }

    pub fn train(&mut self) {

        self.model = Logistic::new(
            &self.x_train, 
            &self.y_train, 
            false, 
            0.00001).unwrap(); 

        self.model.train_batch(10, 32, 1000);
        self.model.save("titanic").unwrap(); 
        println!("Model loss after training: {:?}", self.model.loss());
    }

}


