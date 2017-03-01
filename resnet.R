library(mxnet)
conv_factory <- function(data, num_filter, kernel, stride,
                         pad, act_type = 'relu', conv_type = 0) {
  if (conv_type == 0) {
    conv = mx.symbol.Convolution(data = data, num_filter = num_filter,
                                 kernel = kernel, stride = stride, pad = pad)
    bn = mx.symbol.BatchNorm(data = conv)
    act = mx.symbol.Activation(data = bn, act_type = act_type)
    return(act)
  } else if (conv_type == 1) {
    conv = mx.symbol.Convolution(data = data, num_filter = num_filter,
                                 kernel = kernel, stride = stride, pad = pad)
    bn = mx.symbol.BatchNorm(data = conv)
    return(bn)
  }
}
residual_factory <- function(data, num_filter, dim_match) {
  if (dim_match) {
    identity_data = data
    conv1 = conv_factory(data = data, num_filter = num_filter, kernel = c(3, 3),
                         stride = c(1, 1), pad = c(1, 1), act_type = 'relu', conv_type = 0)
    
    conv2 = conv_factory(data = conv1, num_filter = num_filter, kernel = c(3, 3),
                         stride = c(1, 1), pad = c(1, 1), conv_type = 1)
    new_data = identity_data + conv2
    act = mx.symbol.Activation(data = new_data, act_type = 'relu')
    return(act)
  } else {
    conv1 = conv_factory(data = data, num_filter = num_filter, kernel = c(3, 3),
                         stride = c(2, 2), pad = c(1, 1), act_type = 'relu', conv_type = 0)
    conv2 = conv_factory(data = conv1, num_filter = num_filter, kernel = c(3, 3),
                         stride = c(1, 1), pad = c(1, 1), conv_type = 1)
    
    # adopt project method in the paper when dimension increased
    project_data = conv_factory(data = data, num_filter = num_filter, kernel = c(1, 1),
                                stride = c(2, 2), pad = c(0, 0), conv_type = 1)
    new_data = project_data + conv2
    act = mx.symbol.Activation(data = new_data, act_type = 'relu')
    return(act)
  }
}

residual_net <- function(data, n) {
  #fisrt 2n layers
  for (i in 1:n) {
    data = residual_factory(data = data, num_filter = 16, dim_match = TRUE)
  }
  
  
  #second 2n layers
  for (i in 1:n) {
    if (i == 1) {
      data = residual_factory(data = data, num_filter = 32, dim_match = FALSE)
    } else {
      data = residual_factory(data = data, num_filter = 32, dim_match = TRUE)
    }
  }
  #third 2n layers
  for (i in 1:n) {
    if (i == 1) {
      data = residual_factory(data = data, num_filter = 64, dim_match = FALSE)
    } else {
      data = residual_factory(data = data, num_filter = 64, dim_match = TRUE)
    }
  }
  return(data)
}
get_symbol <- function(num_classes = 10) {
  conv <- conv_factory(data = mx.symbol.Variable(name = 'data'), num_filter = 16,
                       kernel = c(3, 3), stride = c(1, 1), pad = c(1, 1),
                       act_type = 'relu', conv_type = 0)
  n <- 3 # set n = 3 means get a model with 3*6+2=20 layers, set n = 9 means 9*6+2=56 layers
  resnet <- residual_net(conv, n) #
  pool <- mx.symbol.Pooling(data = resnet, kernel = c(7, 7), pool_type = 'max')
  flatten <- mx.symbol.Flatten(data = pool, name = 'flatten')
  fc <- mx.symbol.FullyConnected(data = flatten, num_hidden = num_classes, name = 'fc1')
  softmax <- mx.symbol.SoftmaxOutput(data = fc, name = 'softmax')
  return(softmax)
}
datf<-df[sample(ncol(df), ncol(df), )]
train<-datf
train_y<-train$label
train$label<-NULL
train <- data.matrix(train)
train_y<data.matrix(train_y)
str(train)
train_x<-t(-train_y)
train_array <- train_x
dim(train_array) <- c(200, 200, 1, ncol(train_x))

test_x <- t(test[, -1])
test_y <- test[, 1]
test_array <- test_x
dim(test_array) <- c(200, 200, 1, ncol(test_x))
devices <- mx.cpu()
res1 <- get_symbol()
mx.set.seed(0)
model <- mx.model.FeedForward.create(res1,
                                     X = train_array,
                                     y = train_y,
                                     eval.data = list(data = test_array, label = test_y),
                                     ctx = devices,
                                     num.round = 10, # number of epochs
                                     array.batch.size = 40,
                                     learning.rate = 0.01,
                                     momentum = 0.5,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))



data <- data.frame(A = c("egg", "egg"), B = c(NA, "bacon"), C = c("ham", "ham"), D = c(NA, NA))
keys <- lapply(data, function(x) if(is.factor(x)) levels(x) else "bacon")
vals <- names(data)
lookup <- setNames(vals,keys)
lookup
