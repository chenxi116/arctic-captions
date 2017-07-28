"""
Sampling script for attention models

Works on CPU with support for multi-process
"""
import argparse
import numpy
from scipy.io import savemat
import cPickle as pkl
import pdb

from capgen import build_sampler, gen_sample, \
                   load_params, \
                   init_params, \
                   init_tparams, \
                   get_dataset

from multiprocessing import Process, Queue


# single instance of a sampling process
def gen_model(queue, rqueue, pid, model, options, k, normalize, word_idict, sampling):
    import theano
    from theano import tensor
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

    trng = RandomStreams(1234)
    # this is zero indicate we are not using dropout in the graph
    use_noise = theano.shared(numpy.float32(0.), name='use_noise')

    # get the parameters
    params = init_params(options)
    params = load_params(model, params)
    tparams = init_tparams(params)

    # build the sampling computational graph
    # see capgen.py for more detailed explanations
    f_init, f_next = build_sampler(tparams, options, use_noise, trng, sampling=sampling)

    def _gencap(cc0, cc1=0):
        sample, score, alphas, alpha_sample = gen_sample(tparams, f_init, f_next, cc0, options,
                                   trng=trng, k=k, maxlen=200, stochastic=False, gt_cap = cc1)
        # adjust for length bias
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx], alphas

    while True:
        req = queue.get()
        # exit signal
        if req is None:
            break

        idx, cap, context = req[0], req[1], req[2]
        print pid, '-', idx
        seq, alphas = _gencap(context, 0) # change cap to 0 for old code
        rqueue.put((idx, seq, alphas))

    return 

def main(model, saveto, k=5, normalize=False, zero_pad=False, n_process=5, datasets='dev,test', sampling=False, pkl_name=None, ic=0):
    # load model model_options
    if pkl_name is None:
        pkl_name = model
    with open('%s.pkl'% pkl_name, 'rb') as f:
        options = pkl.load(f)

    # fetch data, skip ones we aren't using to save time
    load_data, prepare_data = get_dataset(options['dataset'])
    _, valid, test, worddict = load_data(load_train=False, load_dev=True if 'dev' in datasets else False,
                                             load_test=True if 'test' in datasets else False)

    # <eos> means end of sequence (aka periods), UNK means unknown
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # create processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(target=gen_model, 
                                  args=(queue,rqueue,midx,model,options,k,normalize,word_idict, sampling))
        processes[midx].start()

    # index -> words
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict[w])
            capsw.append(' '.join(ww))
        return capsw

    # unsparsify, reshape, and queue
    def _send_jobs(caps, contexts, worddict, ic):
        for idx, ctx in enumerate(contexts):
            cc = caps[idx*5 + ic]
            seqs = [worddict[w] if w in worddict and worddict[w] < 10000 else 1 for w in cc[0].split()]

            cc = ctx.todense().reshape([14*14,512])
            if zero_pad:
                cc0 = numpy.zeros((cc.shape[0]+1, cc.shape[1])).astype('float32')
                cc0[:-1,:] = cc
            else:
                cc0 = cc
            queue.put((idx, seqs, cc0))

    # retrieve caption from process
    def _retrieve_jobs(n_samples):
        caps = [None] * n_samples
        alphas = [None] * n_samples
        for idx in xrange(n_samples):
            resp = rqueue.get()
            caps[resp[0]] = resp[1]
            alphas[resp[0]] = resp[2]
            if numpy.mod(idx, 10) == 0:
                print 'Sample ', (idx+1), '/', n_samples, ' Done'
        return caps, alphas

    ds = datasets.strip().split(',')

    # send all the features for the various datasets
    for dd in ds:
        if dd == 'dev':
            print 'Development Set...',
            _send_jobs(valid[0], valid[1], worddict, ic)
            capsi, alphas = _retrieve_jobs(len(valid[1].todense()))
            caps = _seqs2words(capsi)
            with open(saveto+'.dev.txt', 'w') as f:
                print >>f, '\n'.join(caps)
            # pkl.dump(alphas, open(saveto+'.alphas.dev.pkl', 'wb'))
            savemat(saveto+'.alphas.dev.mat', mdict = {'alphas': alphas})
            print 'Done'
        if dd == 'test':
            print 'Test Set...',
            _send_jobs(test[0], test[1], worddict, ic)
            capsi, alphas = _retrieve_jobs(len(test[1].todense()))
            caps = _seqs2words(capsi)
            with open(saveto+'.test.txt', 'w') as f:
                print >>f, '\n'.join(caps)
            # pkl.dump(alphas, open(saveto+'.alphas.test.pkl', 'wb'))
            savemat(saveto+'.alphas.test.mat', mdict = {'alphas': alphas})
            print 'Done'
    # end processes
    for midx in xrange(n_process):
        queue.put(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('-sampling', action="store_true", default=False) # this only matters for hard attention
    parser.add_argument('-p', type=int, default=4, help="number of processes to use")
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-z', action="store_true", default=False)
    parser.add_argument('-d', type=str, default='dev,test')
    parser.add_argument('-pkl_name', type=str, default=None, help="name of pickle file (without the .pkl)")
    parser.add_argument('model', type=str)
    parser.add_argument('saveto', type=str)
    parser.add_argument('-r', type=int, default=0, help="index of reference") # 0, 1, 2, 3, 4

    args = parser.parse_args()
    main(args.model, args.saveto, k=args.k, zero_pad=args.z, pkl_name=args.pkl_name,  n_process=args.p, normalize=args.n, datasets=args.d, sampling=args.sampling, ic=args.r)
