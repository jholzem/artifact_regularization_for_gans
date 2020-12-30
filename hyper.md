# Hyperparameter Tuning: total time budget = 36 h

Maximum number of epochs: 9                                                     (1 value with 9 checkpoints - 3 hours (20 min per epoch))

Learning rate (LR): 1e-3, 1e-4, 1e-5, 1e-6                                      (4 values)

Lambda: 1, 0.5, 0.1, 0.05, 0.01, 0.005                                          (6 values)

Metric: '2', 'cos'                                                              (2 values)


### Division

Amir:   LR: all,    Lambda: 1, 0.5, 0.1,          Metric: '2'

Jonas:  LR: all,    Lambda: 0.05, 0.01, 0.005,    Metric: '2'

Max:    LR: all,    Lambda: 1, 0.5, 0.1,          Metric: 'cos'

Oli:    LR: all,    Lambda: 0.05, 0.01, 0.005,    Metric: 'cos'
