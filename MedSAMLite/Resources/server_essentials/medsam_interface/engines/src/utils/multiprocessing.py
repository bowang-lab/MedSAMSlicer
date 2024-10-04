import multiprocessing

from tqdm import tqdm


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [
        multiprocessing.Process(target=fun, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [
        q_in.put((i, x)) for i, x in enumerate(tqdm(X, position=0, desc="Queue In"))
    ]
    for _ in range(nprocs):
        q_in.put((None, None))
    res = [q_out.get() for _ in tqdm(range(len(sent)), position=1, desc="Queue Out")]
    res = [x for _, x in sorted(res)]
    for p in proc:
        p.join()

    return res
