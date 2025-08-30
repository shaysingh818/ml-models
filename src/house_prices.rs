use ndarray::{s, arr2, Array, Array2, Axis}; 
use polars::prelude::*;
use polars::prelude::ParquetReader;
use dendritic::optimizer::model::*;
use dendritic::optimizer::train::*; 
use dendritic::optimizer::optimizers::*; 
use dendritic::optimizer::regression::sgd::*;
use dendritic::preprocessing::processor::*; 


pub struct HousePrices {

    /// Name of model
    name: String,

    /// Path of where dataset lives
    file_path: String,

    /// Training dataset as ndarray
    x: Array2<f64>,

    /// Target values as ndaarray
    y: Array2<f64>,

    /// Training dataset split
    training_data: (Array2<f64>, Array2<f64>),

    /// Testing data
    testing_data: (Array2<f64>, Array2<f64>),

    /// Model associated with pipeline
    model: SGD,


}


impl ModelPipeline for HousePrices {
 
    fn register(name: &str) -> Self {

        let temp_x: Array2<f64> = Array2::zeros((0, 0));
        let temp_y: Array2<f64> = Array2::zeros((0, 0));

        HousePrices {
            name: name.to_string(),
            file_path: "data/california_housing.parquet".into(),
            x: temp_x.clone(),
            y: temp_y.clone(),
            training_data: (temp_x.clone(), temp_y.clone()),
            testing_data: (temp_x.clone(), temp_y.clone()),
            model: SGD::new(&temp_x, &temp_y, 0.01).unwrap()
        }
    }

}



impl Load for HousePrices {

    fn load(&mut self) {

        println!("Running load step for: {:?}", self.name); 

        let mut file = std::fs::File::open(&self.file_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();

        let df_select = df.select([
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "population",
            "median_income",
        ]).unwrap();


        let df_target = df.select(["median_house_value"]).unwrap(); 

        self.x = df_select.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

        self.y = df_target.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

    }

}


impl Transform for HousePrices {

    fn transform(&mut self) {

        println!("Running transform step for: {:?}", self.name);

        let num_rows = self.x.nrows(); 
        if num_rows != self.y.nrows() {
            panic!("Number of rows for sample features and target unequal");
        }

        let temp_x = self.x.clone();
        let mut scalar = MinMaxScalar::new(&temp_x).unwrap();
        let encoded = scalar.encode();


        let temp_y = self.y.clone();
        let mut scalar_y = MinMaxScalar::new(&temp_y).unwrap();
        let y_encoded = scalar_y.encode();

        self.x = encoded;
        self.y = y_encoded; 

        let train_split = 0.8 * num_rows as f64; 
        let test_split = 0.2 * num_rows as f64;

        self.training_data = (
            self.x.slice(s![0..train_split as usize, ..]).to_owned(), 
            self.y.slice(s![0..train_split as usize, ..]).to_owned()
        );

        self.testing_data = (
            self.x.slice(
                s![train_split as usize..num_rows, ..]
            ).to_owned(), 
            self.y.slice(
                s![train_split as usize..num_rows, ..]
            ).to_owned()
        );


    }

}


impl Train for HousePrices {

    fn train(&mut self) {

        println!("Running train step for: {:?}", self.name); 

        self.model = SGD::new(
            &self.training_data.0, 
            &self.training_data.1, 
            0.0001
        ).unwrap();

        //let mut opt = Nesterov::default(&self.model);

        self.model.train_batch(5, 128, 1000);
        self.model.save("models/housing_prices").unwrap();

    }

}


impl Inference for HousePrices {

    fn inference(&mut self) {

        println!("Running inference step for: {:?}", self.name); 

        let mut loaded = SGD::load("models/housing_prices").unwrap();

        let x1 = self.testing_data.0.slice(s![0..5, ..]);
        let y1 = self.testing_data.1.slice(s![0..5, ..]);

        println!("First set of predictions");
        println!("{:?}", y1);
        println!(""); 
        println!("{:?}", loaded.predict(&x1.to_owned())); 

        let x2 = self.testing_data.0.slice(s![50..60, ..]);
        let y2 = self.testing_data.1.slice(s![50..60, ..]);

        println!("Second set of predictions");
        println!("{:?}", y2);
        println!(""); 
        println!("{:?}", loaded.predict(&x2.to_owned())); 

        let x3 = self.testing_data.0.slice(s![10..15, ..]);
        let y3 = self.testing_data.1.slice(s![10..15, ..]);


        /*
        let mut scalar = MinMaxScalar::new(&y3.to_owned()).unwrap();
        let decoded = scalar.decode(&y3.to_owned());

        let predicted = loaded.predict(&x3.to_owned());

        let mut scalar2 = MinMaxScalar::new(&predicted).unwrap();

        let decoded2 = scalar2.decode(&predicted);

        println!("Third set of predictions");
        println!("{:?}", decoded);
        println!(""); 
        println!("{:?}", decoded2); */  

    }

}
