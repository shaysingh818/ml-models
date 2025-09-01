use dendritic::optimizer::model::*; 
use dendritic_ml_models::iris::IrisFlowersModel;
use dendritic_ml_models::breast_cancer::BreastCancerModel; 
use dendritic_ml_models::house_prices::HousePrices; 
use dendritic_ml_models::student_performance::StudentPerformance; 

fn main() {

    /*

    let mut model = BreastCancerModel::register("breast_cancer");
    model.load();
    model.transform();
    model.train();
    model.inference();


    let mut model = StudentPerformance::register("student_performance");
    model.load();
    model.transform();
    model.train();
    model.inference(); 

    let mut model = IrisFlowersModel::register("iris_flowers");
    model.load();
    model.transform();
    model.train();
    model.inference(); */

    let mut model = HousePrices::register("housing_prices");
    model.load();
    model.transform();
    model.train();
    model.inference(); 

}
