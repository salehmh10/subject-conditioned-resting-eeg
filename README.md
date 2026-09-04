# Subject-Conditioned Resting-State EEG

Main repository for Scenario 2 of the synthetic EEG project.

## Current status

Scenario 1:
- completed and frozen
- separate repository: `eye-state-eeg-diffusion`
- frozen commit: `2731f3d`
- frozen tag: `scenario1-final`

Scenario 2:
- standalone project
- active development
- local Git commits during development
- no GitHub push until the study is complete

## Main dataset

MPI-LEMON resting-state EEG.

The primary conditions are:

- Eyes Open (EO)
- Eyes Closed (EC)

Raw and processed EEG data are not redistributed in this repository.

## Main scientific questions

1. Can subject- and eye-state-conditioned EEG generation preserve individual EO-to-EC spectral and functional-connectivity changes on held-out recording blocks?

2. Does adding synthetic EEG to real training data improve classification performance on held-out real EEG?

## Planned generative models

- B0: Eye-state-only DDPM
- B1: Subject + Eye-State DDPM
- B2: Subject + Eye-State DDPM + Spectral Regularization

## Planned classifiers

- C1: Log-bandpower + Logistic Regression
- C2: Covariance / Tangent Space + Linear SVM
- C3: EEGNetv4

## Planned augmentation study

Synthetic-to-real ratios:

- 0.5x
- 1.0x
- 2.0x

An IAAFT spectral-surrogate control will also be evaluated.

## Core validation rule

Validation and Test are real-only.

The final Test set will remain locked until preprocessing, models, CFG, classifiers, augmentation ratio, seeds, metrics, and statistical procedures have been finalized.
