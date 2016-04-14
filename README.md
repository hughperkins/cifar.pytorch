# cifar.torch

For original readme, please see https://github.com/szagoruyko/cifar.torch , and Sergey's blog post at http://torch.ch/blog/2015/07/30/cifar.html

This readme is for pytorch version, which handles data loading and preprocessing in python

## Data download

```bash
./download-cifar.sh
```

## Training

```bash
python train.py
```

You should see the loss gradually decrease, and the test accuracy gradually decrease.

# Differences from original lua version

- data loading in python
- preprocessing in python
- no conversion from rgb to yuv (just because... haven't added it)
- no graph for now (but... it's python... you can use all the matplot goodness you are used to using :-) )

