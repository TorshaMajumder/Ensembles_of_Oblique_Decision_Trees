# Ensembles of Oblique Decision Trees
#### Author: Torsha Majumder

## Background
This repository contains several decision tree algorithms compatible with **Scikit-Learn's *Bagging Classifier***. For the complete experimental setup and results, please check [my thesis](https://utd-ir.tdl.org/handle/10735.1/8818). If you find this code useful, please cite my work.

## Citation
@mastersthesis{Majumder2020Ensembles,
  author       = {Majumder, T.},
  title        = {Ensembles of oblique decision trees},
  school       = {University of Texas, Dallas},
  year         = {2020},
  type         = {Master's Thesis},
  note         = {UTD Theses and Dissertations}
}

## Experiment
Decision Trees considered for this experiment:

    * Standard Decision Tree with Bagging
    * Oblique Classifier 1 with Bagging
    * Weighted Oblique Decision Tree with Bagging
    * Randomized CART with Bagging
    * HouseHolder CART with Bagging
    * Continuous Optimization of Oblique Splits with Bagging
    * Deep Neural Decision Trees with Bagging
    * Non-Linear Decision Trees with Bagging
    * Random Forest Classifier


In this experiment we have to skip OC1, DNDT and, NDT classifiers due to its computational cost.


This experiment has been conducted on 12 Benchmark Data sets.

    * Iris                 * Vehicle
    * Wine                 * Fourclass
    * Glass                * Segment
    * Heart                * Satimage
    * Breast-cancer        * Pendigits
    * Diabetes             * Letter







