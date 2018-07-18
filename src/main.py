import argparse
import os
import itertools
import random
import logging
import yaml
import numpy as np
import torch
import shutil
import time
import torch.optim as optim
import ipdb
import viz
import create_video
import matplotlib.pyplot as plt
import forward_kinematics
import data_utils
import model as md
from torch import nn
from torch.autograd import Variable

def get_batch(data, actions ):
    """Get a random batch of data from the specified bucket, prepare for step.

    Args
      data: a list of sequences of size n-by-d to fit the model to.
      actions: a list of the actions we are using
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    # Select entries at random
    all_keys    = list(data.keys())
    chosen_keys = np.random.choice( len(all_keys), args.batch_size )

    # How many frames in total do we need?
    source_seq_len = args.seq_length_in
    target_seq_len = args.seq_length_out
    total_frames = source_seq_len + target_seq_len
    
    if args.omit_one_hot:
        input_size = 54
    else:
        input_size = 54 + 15
    
    encoder_inputs  = np.zeros((args.batch_size, source_seq_len-1,input_size), dtype=float)
    decoder_inputs  = np.zeros((args.batch_size, target_seq_len, input_size), dtype=float)
    decoder_outputs = np.zeros((args.batch_size, target_seq_len, input_size), dtype=float)

    for i in range( args.batch_size ):

      the_key = all_keys[ chosen_keys[i] ]

      # Get the number of frames
      n, _ = data[ the_key ].shape

      # Sample somewherein the middle
      idx = np.random.randint( 16, n-total_frames )

      # Select the data around the sampled points
      data_sel = data[ the_key ][idx:idx+total_frames ,:]

      # Add the data
      encoder_inputs[i,:,0:input_size]  = data_sel[0:source_seq_len-1, :]
      decoder_inputs[i,:,0:input_size]  = data_sel[source_seq_len-1:source_seq_len+target_seq_len-1, :]
      decoder_outputs[i,:,0:input_size] = data_sel[source_seq_len:, 0:input_size]

    return encoder_inputs, decoder_inputs, decoder_outputs

def find_indices_srnn(data, action ):
    """
    Find the same action indices as in SRNN.
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState( SEED )

    subject = 5
    subaction1 = 1
    subaction2 = 2

    T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
    T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
    prefix, suffix = 50, 100

    idx = []
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    return idx


def get_batch_srnn(data, action ):
    """
    Get a random batch of data from the specified bucket, prepare for step.

    Args
      data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
        v=nxd matrix with a sequence of poses
      action: the action to load data from
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """
    if args.omit_one_hot:
        input_size = 54
    else:
        input_size = 54 + 15
    actions = ["directions", "discussion", "eating", "greeting", "phoning",
              "posing", "purchases", "sitting", "sittingdown", "smoking",
              "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

    if not action in actions:
      raise ValueError("Unrecognized action {0}".format(action))

    frames = {}
    frames[ action ] = find_indices_srnn( data, action )

    batch_size = 8 # we always evaluate 8 seeds
    subject    = 5 # we always evaluate on subject 5
    source_seq_len = args.seq_length_in
    target_seq_len = args.seq_length_out

    seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

    #encoder_inputs  = np.zeros( (batch_size, source_seq_len, input_size), dtype=float )
    encoder_inputs  = np.zeros( (batch_size, source_seq_len-1, input_size), dtype=float )
    decoder_inputs  = np.zeros( (batch_size, target_seq_len, input_size), dtype=float )
    decoder_outputs = np.zeros( (batch_size, target_seq_len, input_size), dtype=float )

    # Compute the number of frames needed
    total_frames = source_seq_len + target_seq_len

    # Reproducing SRNN's sequence subsequence selection as done in
    # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    for i in range( batch_size ):

      _, subsequence, idx = seeds[i]
      idx = idx + 50

      data_sel = data[ (subject, action, subsequence, 'even') ]

      data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]
      encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
      decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :] #49~99
      decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]                                    #50~100


    return encoder_inputs, decoder_inputs, decoder_outputs
    
    
    
    
    
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.base_lr * (0.1 ** (epoch // args.lr_epochs))
    print('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr 
    
def read_all_data( actions, seq_length_in, seq_length_out, data_dir, one_hot ):
  """
  Loads data for training/testing and normalizes it.

  Args
    actions: list of strings (actions) to load
    seq_length_in: number of frames to use in the burn-in sequence
    seq_length_out: number of frames to use in the output sequence
    data_dir: directory to load the data from
    one_hot: whether to use one-hot encoding per action
  Returns
    train_set: dictionary with normalized training data
    test_set: dictionary with test data
    data_mean: d-long vector with the mean of the training data
    data_std: d-long vector with the standard dev of the training data
    dim_to_ignore: dimensions that are not used becaused stdev is too small
    dim_to_use: dimensions that we are actually using in the model
  """

  # === Read training data ===
  print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
           seq_length_in, seq_length_out))

  train_subject_ids = [1,6,7,8,9,11]
  test_subject_ids = [5]

  train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions, one_hot )
  test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions, one_hot )

  # Compute normalization stats
  data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

  # Normalize -- subtract mean, divide by stdev
  train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
  test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
  print("done reading data.")

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

def get_srnn_gts( actions, test_set, data_mean, data_std, dim_to_ignore, one_hot, to_euler=True ):
  """
  Get the ground truths for srnn's sequences, and convert to Euler angles.
  (the error is always computed in Euler angles).

  Args
    actions: a list of actions to get ground truths for.
    model: training model we are using (we only use the "get_batch" method).
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map

  Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
  """
  srnn_gts_euler = {}

  for action in actions:

    srnn_gt_euler = []
    _, _, srnn_expmap = get_batch_srnn( test_set, action )

    # expmap -> rotmat -> euler
    for i in np.arange( srnn_expmap.shape[0] ):
      denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          for k in np.arange(3,97,3):
            denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

      srnn_gt_euler.append( denormed );

    # Put back in the dictionary
    srnn_gts_euler[action] = srnn_gt_euler

  return srnn_gts_euler

def visual2(action1, action2):
    n,m = action1.shape
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    ax =  fig.add_subplot(1, 2, 1, projection='3d')
    ax2 =  fig.add_subplot(1, 2, 2,  projection='3d')
    ob = viz.Ax3DPose(ax)
    ob2 = viz.Ax3DPose(ax2)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    for i in range(n):
        ob.update( action1[i,:] )
       
        ob2.update( action2[i,:] )
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.001)
    plt.close()
def visual(action_sequence):
    n,m = action_sequence.shape
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ob = viz.Ax3DPose(ax)
    for i in range(n):
        ob.update( action_sequence[i,:] )
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.001)
        #filename = '{}/{}.png'.format( '/home/xingyu/output', i)
        #set_trace()
        #plt.savefig(filename)
    plt.close()
def define_actions( action ):
  """
  Define the list of actions we are using.

  Args
    action: String with the passed action. Could be "all"
  Returns
    actions: List of strings of actions
  Raises
    ValueError if the action is not included in H3.6M
  """

  actions = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]

  if action in actions:
    return [action]

  if action == "all":
    return actions

  if action == "all_srnn":
    return ["walking", "eating", "smoking", "discussion"]

  raise( ValueError, "Unrecognized action: %d" % action )

def train():
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables()
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
        actions, args.seq_length_in, args.seq_length_out, args.data_dir, not args.omit_one_hot )
        
    srnn_gts_euler = get_srnn_gts( actions, test_set, data_mean,
                              data_std, dim_to_ignore, not args.omit_one_hot )
    current_step = 0
    previous_losses = []
    step_time, loss = 0, 0 

    
    for _ in range(args.iterations):
        model.train()
        start_time = time.time()

        encoder_inputs, decoder_inputs, decoder_outputs = get_batch( train_set, not args.omit_one_hot )

        batch_data = torch.Tensor(encoder_inputs)
        target = torch.Tensor(decoder_outputs)

        
        if args.cuda:
            batch_data = batch_data.cuda()
            target = target.cuda()
        batch_data = Variable(batch_data.float())  #batch, input_size, seq_len
        optimizer.zero_grad()  # Clear the gradients of all optimized variables 
        batch_data = batch_data 
        output1 = model(batch_data)
        target = target.float()
        loss = MSE_criterion(output1, target)
        loss.backward()
        current_step += 1
        optimizer.step()
 
        if current_step % 10== 0:
            print('iteration: {}/{} reg_loss = {},'.format(current_step, args.iterations, loss.data[0]))
        if current_step % args.test_every == 0:
            model.eval()
            '''
            encoder_inputs, decoder_inputs, decoder_outputs = get_batch( test_set, not args.omit_one_hot )
            batch_data = torch.Tensor(encoder_inputs)
            target = torch.Tensor(decoder_outputs)
     
        
            if args.cuda:
                batch_data = batch_data.cuda()
                target = target.cuda()
            batch_data = Variable(batch_data.float())  #batch, input_size, seq_len
            optimizer.zero_grad()  # Clear the gradients of all optimized variables 
            batch_data = batch_data 
            output1 = model(batch_data)
            target = target.float()
            loss = MSE_criterion(output1, target)
            '''
            print()
            print("{0: <16} |".format("milliseconds"), end="")
            for ms in [80, 160, 320, 400, 560, 1000]:
                print(" {0:5d} |".format(ms), end="")
            print()
            
            for action in actions:
                encoder_inputs, decoder_inputs, decoder_outputs = get_batch_srnn( test_set, action )
                batch_data = torch.Tensor(encoder_inputs)
                target = torch.Tensor(decoder_outputs)
     
        
                if args.cuda:
                    batch_data = batch_data.cuda()
                    target = target.cuda()
                batch_data = Variable(batch_data.float())  #batch, input_size, seq_len
                optimizer.zero_grad()  # Clear the gradients of all optimized variables 
        
                output1 = model(batch_data)
                target = target.float()
                loss = MSE_criterion(output1, target)
               
                
                srnn_pred_expmap = data_utils.revert_output_format( output1,
                        data_mean, data_std, dim_to_ignore, actions, not args.omit_one_hot )
                mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )      
                N_SEQUENCE_TEST = 8 
         
                for i in np.arange(N_SEQUENCE_TEST):
                    eulerchannels_pred = srnn_pred_expmap[i]

                # Convert from exponential map to Euler angles
                    for j in np.arange( eulerchannels_pred.shape[0] ):
                        for k in np.arange(3,97,3):
                            eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

                    gt_i=np.copy(srnn_gts_euler[action][i])
                    gt_i[:,0:6] = 0

                    idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]
     
                    euc_error = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
                    euc_error = np.sum(euc_error, 1)
                    euc_error = np.sqrt( euc_error )
                    mean_errors[i,:] = euc_error
                mean_mean_errors = np.mean( mean_errors, 0 )
                print("{0: <16} |".format(action), end="")
                for ms in [1,3,7,9,13,24]:
                    if args.seq_length_out >= ms+1:
                        print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
                    else:
                        print("   n/a |", end="")
                print()
         
            action_sequence = np.zeros([args.seq_length_out, 96])
            action_sequence_gt = np.zeros([args.seq_length_out, 96])
            for i in range(args.seq_length_out):
                action_sequence[i,:] = forward_kinematics.fkl(eulerchannels_pred[i,:], parent, offset, rotInd, expmapInd)
                action_sequence_gt[i,:] = forward_kinematics.fkl(gt_i[i,:], parent, offset, rotInd, expmapInd)
            visual2(action_sequence, action_sequence_gt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default= 100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_length_in', type=int, default=49)
    parser.add_argument('--seq_length_out', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=54)
    parser.add_argument('--output_size', type=int, default=54)
    parser.add_argument('--hidden_size', type=int, default=256)
  
  
    
    parser.add_argument('--istrain', default = True)
    parser.add_argument('--checkpoint', default= 50)
    parser.add_argument('--res_dir', type=str, default='./model')
    parser.add_argument('--epoch', type=int, default=200)
    
    parser.add_argument('--action', default = 'all')
    
    parser.add_argument('--iterations', default = 100000)
    parser.add_argument('--test_every', default = 1000)
    parser.add_argument('--save_every', default = 100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--omit_one_hot', default = True, help='Whether to remove one-hot encoding from the data')
    parser.add_argument('--data_dir', default = os.path.normpath("../data/h3.6m/dataset"))
   
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--initialization', type=str, default='default')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--lr-epochs', type=int, default=200, metavar='N',
            help='number of epochs to decay the lr (default: 15)')
 
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
  
    best_prec1 = 0 
    
    if args.out_dir == '':
        out_dir = os.path.join(args.res_dir, 'model')
    else:
        out_dir = args.out_dir
    
    
    actions = define_actions(args.action) 
    train_subjects = [1,6,7,8,9,11]
    test_subjects = [5] 
    


    
    if args.istrain:
        model = md.LSTMRegressPose(args.input_size, args.hidden_size, args.output_size, args.seq_length_in, args.seq_length_out)
    else:
        model = md.LSTMRegressPose(args.input_size, args.hidden_size, args.output_size, args.seq_length_in, args.seq_length_out)
        
        load_path = os.path.join(args.res_dir, 'Checkpoint_{}_.pth.tar'.format(args.checkpoint))
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['state_dict'])
    
    if args.cuda:
        model.cuda()
        print(model)
        
    param_dict = dict(model.named_parameters())
    params = []
    
    base_lr = args.base_lr
    
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': args.base_lr,
            'weight_decay': args.weight_decay,
            'key':key}]
    
    optimizer = optim.Adam(params, lr=args.base_lr,
            weight_decay=args.weight_decay)

    MSE_criterion = nn.MSELoss()

    
    if args.istrain:
        for epoch in range(1, args.epoch + 1):
            adjust_learning_rate(optimizer, epoch)
          
            train()
            train(epoch)
        
            '''
            prec1 = validate()
        
        
            if args.cuda:
                is_best = prec1.cpu().data.numpy().tolist() > best_prec1
                best_prec1 = max(prec1.cpu().data.numpy().tolist(), best_prec1)
            else:
                is_best = prec1.data.numpy()[0] > best_prec1
                best_prec1 = max(prec1.data.numpy()[0], best_prec1)
            '''
            file_ = 'Checkpoint_{}_.pth.tar'.format(epoch)
            file_name = os.path.join(args.res_dir,file_)
            is_best = False
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                },is_best, file_name)  
    else:
         prec1 = test()
    