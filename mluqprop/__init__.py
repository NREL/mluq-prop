"""MLUQ-PROP"""
import os

MLUQPROP_DIR = os.path.dirname(os.path.realpath(__file__))
MLUQPROP_BNN_DIR = os.path.join(MLUQPROP_DIR, "BNN")
MLUQPROP_OOD_DIR = os.path.join(MLUQPROP_DIR, "OOD")
MLUQPROP_DIMRED_DIR = os.path.join(MLUQPROP_DIR, "dimensionReduction")
MLUQPROP_DATA_DIR = os.path.join(MLUQPROP_DIR, "../data")
MLUQPROP_MODEL_DIR = os.path.join(MLUQPROP_DIR, "../models")
