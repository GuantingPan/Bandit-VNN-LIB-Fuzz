for ((it=1;it<=10;it++)) do
    random_fuzzer -dnn $it.onnx -p $it.vnnlib -nrng 15
done
