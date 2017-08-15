#!/usr/bin/env bash
KERAS_BACKEND="theano" THEANO_FLAGS=device=gpu,floatX=float32 python train_xor_lstm.py
