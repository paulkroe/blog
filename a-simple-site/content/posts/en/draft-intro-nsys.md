---
date: 2026-04-27T14:25:03-04:00
# description: ""
# image: ""
lastmod: 2026-04-27
showTableOfContents: true
tags: ["high-perf-ml","ml"]
title: "A Brief Introduction to GPU Performance Optimization with Nsight Systems"
type: "post"
---

## Introduction
tldr; utilizing full hardware performance is important in modern ml systems. there are many tools to profile and optimize hardware performance. in this post we focus on using nsight systems to optimize the utilization of nvidia gpus. we are intentionally working with a real world example and not some toy examples

Arguably, a lot of ideas in modern machine learning are actually that modern. they data back 40/50 year back (examples and references here). the reason machine learning took off in the last decode or two is that we needed hardware to catch up to enable us to actually implement these ideas. it feels silly to say but it really does make a difference if your training epoch takes 1 second versus 1 hour. while this observation seems obvious in hindsight, in some shape or for the same observation is still true (even with morse law having 50 years of doing its magic).
instead of putting together a collection of toy examples we will work with a real world production codebase. A) because i have to optimize its performance anyways and B) because that is way more meaningful and hopeful helps you to learn the workflow of optimizing more generally and not overfit on a bunch of toy examples.

## profiling your first program
very briefly, here is how profiling with nsys systems works. assume you have program that you want to profile. for example you might run a training run like this on your training/remote machine:

```
python train.py --config configs/default.yaml
```

assuming you have everything installed you should be able to just run:
```
nsys profile --python-sampling=true --backtrace=none --trace=cuda,nvtx,osrt -o profiling/report --force-overwrite true python train.py --config configs/default.yaml
```
you don't need most of the commandline args above, but i find them useful and usually default to that combo. here is a brief on what they actually do:
- TODO


now you can download the report that has been generated and run this command on your local machine (doesn't need a nvida gpu but needs a GUI)

```
nsys-ui report.nsys-rep 
```

this should open a report that looks somewhat like this:
<img src="/images/intro-nsys/overview.png" alt="Nsight Systems' Overview over Nsight Systems GUI" width="500"/>  

before we jump into the details here, very briefly what you see here:
you can leftclick select regions that seem to be interesting and press shift+z to zoom in. i you want to undo the zoom you can do backspace. if some part of the program is interesting to you, try hovering your point over it and it will display an overlay with additional information that can be very useful.

the top row you see is your device (i.e. your gpu). it shows both its sm utilization (i.e. how good you are utilizing its compute) and its memory utiization (i.e. how much of the gpu's memory you are using). as the gpu is probably the most expensive single part of your system, you generally want these to be high (for memory the picture is somewhat more nuanced but for sm/compute utilization you want this to be quite high all the time). below the top row that is for the gpu you see the section for your cpu (threads 36 means that i have 36 threads running.) the thread named [90759] python is the main thread. in most of the cases you want to pay close attention to what the main thread is doing and can ignore the other threads. should you ever wonder what these other threads are doing you cna click on the plus sign in the bottom left to expand hidden threads. the `--python-sampling=true` flag in our profiling command makes it so that nsys regularly captures the callstack of the running python program. if you hover your mouse above one such time stamp you can see what your cpu is doing. that is really helpful for finding unnecessary synconization point in your code with we will make ues of later on. 
<img src="/images/intro-nsys/hover.png" alt="Python call stack of main thread." width="500"/>  
next you can see the system calls (i.e. what your program is asking your operating system to do.) often useful for understanding when your data is loaded.
finally, you see the cuda api. these are similar to system calls in the sense that you program asks another system to do something for it, but in this case it doesn't as your operating system, but your gpu. we use this frequently to understand when our program is syncing with the gpu (gpu syncs are displayed in green). it also is a nice way to understand when (and ideally why) your gpu is allocating additional memory.

now that you understand how to navigate the tool and understand what you are seeing, lets try to get make things go brrrr.

## nvtx annotations
remember that we said profiling with nsight systems works without modifying your program (as opposed to for example the pytorch profiler).
that is true but still changing your code can help you better understand what is going on. namely, we want to use annotations. these annotaitons can deliniate the functional sections in our programs. i use boiler plate code for nvtx annotations based on the context manager found [here](https://paulbridger.com/posts/nsight-systems-systematic-optimization/)
```
import contextlib
import os
import torch
import functools

def nvtx_annotate(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with nvtx_range(f"{self.__class__.__name__}.{fn.__name__}"):
            return fn(self, *args, **kwargs)
    return wrapper

@contextlib.contextmanager
def nvtx_range(msg: str):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()
```
i usually put that in something like src/profiling.py so that i can import these from anywhere in my project. 
Important:
NVTX Ranges on Host and Device May Not Match, but Both Are Correct

The host submits work via a command queue to the GPU for execution. Due to this asynchronous relationship the start of an NVTX range in the host process will often be well before the start of that range on the GPU.

lets start by adding some annotations like this:
```
    @nvtx_annotate
    def _train_epoch(self, epoch: int) -> tuple[float, LossBreakdown]:
        self.model.train()
        total_loss = 0.0
        last_breakdown = None
        n_samples = 0

        it = getattr(self, "_next_train_iter", None) or iter(self.train_loader)
        self._next_train_iter = None

        n_batches = len(self.train_loader)
        for i, batch in enumerate(it):
            pc = batch["point_cloud"].to(self.device, non_blocking=True)
            B = pc.shape[0]

            self.optimizer.zero_grad()
            with (torch.autocast(device_type=self.device.type) if self.config.training.autocast else contextlib.nullcontext()):
                output = self.model(pc)
                loss, breakdown = self.loss_fn(output, batch, epoch=epoch)

            if i == n_batches - 1:
                # Last batch: reset workers now so the CPU-side handshake and
                # new shuffle index dispatch overlap with the GPU backward pass.
                self._next_train_iter = iter(self.train_loader)

            with nvtx_range("loss.backward"):
                loss.backward()
            with nvtx_range("optimizer.step"):
                self.optimizer.step()

            total_loss += loss.detach().item() * B
            n_samples += B
            last_breakdown = breakdown

            if i == n_batches - 1:
                break

        return total_loss / max(n_samples, 1), last_breakdown
```


this will probably look quite different for your setup, but likely somewhat similar. note that nvtx_range can be used anywhere. the way i set up nvtx_annotate, it only works for annotating class methods. after sprinkling around these tags through the relevant parts of my code i got something like this(note that my annotations cover pretty much the whole exectution. that is important because if we were missing some larger chunks that would mean that there are parts of the code doing a lot of work that we don't mointor and thus can't optimze).
after profiling and opening the report you should now see the nvtx tags. you will see the annotations for the gpu and all threads. as we said, these don't have to align but both are correct and indicative of what a given part of the system is doing at a given moment.

<img src="/images/intro-nsys/nvtx.png" alt="NVTX Annotations" width="500"/>  

appar from the nvtx tags you should also see that gpu utilziations is quite spiky and pretty bad in general. so there is seems to be quite a bit of potential for improvement. if found a similar picture in quite a lot of research codebases that i worked with.

lets start taking a look at this. it makes sense to not start at the beginning because the are a lot of initialization things happening that might take some time. its better to look at the middle section once everything is groved in. what we discuss below is somewhat specific to the code that i am working on. but i want to show you the process more than the specific optimzations in an attempt to make this generalizable to your problems. lets zoom in on some middle sectin (shoft + z).

actually, one sec before we start. we amostly forgot the most important part. we have to track the metric we want to optimize. this should probaly be some mearue of throughput. i usually just measure num_samples / epoch_time. this is nice because it accounts for things like dataset size and batch size. for my code the measurement looks like this.
```
# ---- Training epoch ----
t0 = time.perf_counter()
train_loss, train_breakdown, n_samples = self._train_epoch(epoch)
# a .item() inside _train_epoch already syncs, but in your setup you might want be explicit about syncing
torch.cuda.synchronize()
epoch_time = time.perf_counter() - t0
throughput = n_samples / epoch_time
```
just make sure that at the your computation is actually synced. you don't want to move all computation outside of your timed section instead of actually optimizing it. in my case training stabilizes around roughly 65 samples per second in our initial version (this number will depend a lot on your data, model, loss and hardware. i just report that number to keep track of progress). the general idea is look at section that take long, figure out why they take long and make them faster. pretty easy right?

## dataloader
loading data might be one of the most improtant bottlenecks for you. it can happen quite easily that your gpu spends a lot of time waiting for data. mixed precision can help with that to some extend (more on that later) but there are ways that you can be smart about when and where to load your data. there is a lot of crazy (beautful) engineering that people put into dataloading but luckly for us, pytorch has a few tricks build in for you already. 
increasing number of workers in your dataload
pinning memory
.to(non_blocking=True)

these changes took 5 minutes to through in there and we went from 65 to 75 samples per second. that is already a pretty good improvement for doing almost nothing.

# please use library functions if possible
zooming in on this forward pass we see that there is this section where the gpu does nothing. it neatly aligns with something marked as "Tensor.farthest_point_sample" taking 187ms.

<img src="/images/intro-nsys/fsp.png" alt="Profiling Furthest Point Sampling" width="500"/> 

looking at the corresponding section in the code, reveals a custom fsp implementaiton. there is nothing inhernelty bad about the specific implementation but replacing this with torch.cluster's implementation might be worth a shot here. 

indeed, we jump from 75 samples per second to 140. that is already almost 2x. I didn't to much other than installing torch cluster and replace 10 lines of code. the question is why the author didn't use torch cluster in the first place. my guess would be either A) didn't want to install the dependency, B) claude code/codex didn't want to install the dependency C) the author didn't know torch cluster had an fsp implementation. even though the example looks a bit contrived, i have seen it quite often. i guess the takeway might be if you come across a function that takes quite some time and looks a lot like there might be a library implementation for it, got and look for the library function and replace whatever version you had. below you can see that we fsp section is much faster now.

<img src="/images/intro-nsys/fsp-torch-cluster.png" alt="Profiling Furthest Point Sampling (torch-cluster version)" width="500"/> 


## you really don't want to sync your gpu
Now its time to talk about syncronization. grossly simplified, the cuda compute model works like this. your cpu runs into a massive computation that could be parallized. instead of doing that computation, it launches a kernel to send the computation to the gpu. but of course you don't want your cpu sitting there and waiting whhile the gpu does all the work. so instead of busy waiting your cpu will launch the kernel asyncronously, meaning it will launch the kernel and immediately return to exectuing its next instruction. the hope is that your cpu actually doesn't need the result of the computation immeduately and do other things while waiting for the result. however, once it needs the result, it needs to sync with your gpu to make sure that the results are there. syncing, at least in our simplified version, means that the cpu waits until all computations that have been scheduled on the gpu have been exectued and the results are ready. there are definitively points in your code where your cpu and gpu need to be in sync (for example: ...). but ideally you don't want to sync at some random point in the code waiting for your gpu to finish (for exmaple) (we need to explain why we don't want random syncing here. someone might say: "at the end of the day all computation needs to be synced somewhere so why do we pay attention to this. one "early" sync might help the final sync to be faster." we need to explain that the gpu things a lot about how to schedule computation efficinetly and that these sync points break that (there is more to it and we should expand on that)). it can be hard to tell where syncs are necessary and where they arn't but functions you regard as "utility or nucance" (i.e. functions that you usually don't think about and are happy to offload to some ai agent to implement), having a huge cudaStreamSynconize block in their cuda api row is definitvely a red flag (i don't know this code that well but i dont' think build_target_masks should take that long). to understand what is happending you can look at the stack trace of the python program and hover over it to see what line is called during that massive cuda sync.
<img src="/images/intro-nsys/sync.png" alt="Finding Syncronization Points" width="500"/> 
for me it was something like this
```
n_strokes_per = (group_idx_sorted.max(dim=-1).values + 1).clamp(min=0)  # (B,)
max_n = max(int(n_strokes_per.max().item()), 1)  
```
and it makes sense. `n_strokes_per` lives on the gpu and for us to know what its max is we need to know its result and thus need to sync with the gpu.
calls like this can be hard to remove and it might not be possible to remove them at all. but its definitively worth trying. for my case I was able to change the preprocessing and colating function so that this esentailly became a constant, removing the need to sync here.
i.e. i now have something like this:
```
max_n = n_masks
```

and the performance improvement is .... nothing. doning something really smart and getting 0 performance out of it is part of the process i guess. profiling once more the good news is that the cpu runs through build_target_masks really quickly now. you can bearly see it at the resolution of the image below.

<img src="/images/intro-nsys/sync-two.png" alt="Finding Syncronization Points (part two)" width="500"/> 

the bad news is that there is still this huge sync call in compute_mask_loss. looking at the python backtrace one more we can find the line causing the issue.
the line in question is this if statment:
```
if not stroke_valid.any():
        return pred_mask_logits.sum() * 0.0 # keep grad graph alive, zero
```
the cpu needs to wait on stroke_valid to check whether it should branch or not. again, after thinking about where this was coming from claude & I realized that this was an artifact checking the quality of the data of the current batch. by cleaning the data and checking the condition during preprocessing i could get rid of that part of the code, at least removing it from the hot section.

i solved issues like this probably 4 or 5 times but i won't bore you with the details of a codebase that you are unfaimlar with and have no access to. bottom line is the hot part of my code is now mustly free of unnecessary syncs (i think). we are now ~160 samples/s (up from 140samples/s that is a 15% improvement not too bad).

as i said it might be a bit hard to understand what syncs are really necessary and it definitevely depends on your code but i usually try to understand (at least for the biggest syncs) why they are necessary. 
for me the final (at least for now) picture looks like this:

<img src="/images/intro-nsys/sync-final.png" alt="Final Look at Syncronization" width="500"/> 
(for those that care, here is my working hypothesis for each of the syncs three: the first (large) is from the forward pass through a point encoder that we use for conditioning. we need that result before the rest of the network can condition on that embedding). the second one comes from the fact that we need to compute a matching in the loss. depending on that matching, the loss hsa to look slightly different. the third is at the end of the epoch when we wait for the batch to be finished  (need to be clearer about this last part)).
 

## compiling your model
torch compile. (need to expand on this, what its doing, build to be a on line code change etc.) i like this manual [here](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0). i compiled my nn.module and the class that computes the loss. this brings us up to 170 samples/sec

## mixed precision
torch autocast. same with compiling. need to expand on what its doing. should be easy to add. 
240 samples/s (pretty huge)

## misc
finally i wanted to point out that the system initually needed to compute a matching in the loss. it was doing  that with hungarian matching. people tried implementing hungarian matching for that but it didn't make its way into default pytorch (i think) and you would need to worry about custom cuda kernels for [that](https://github.com/paclopes/HungarianGPU) to make it efficient. instead i used sinkhorn distance. its a commonly used loss function that instead of finding an exact matching finds a soft assignment. its much easier to parallelize. for my usecase it was a drop in replacement that reduced the forward pass through the loss by rouhgly 60%. it was the first thing i changed and probably the single biggest improvement in the whole optimization. the reason that i didn't include it more prominently is that it probably won't generalize as easily to whatever you are working on. but there still is a more general point, namely that its often justified to simplify the loss, even if it lacks the theoretical foundation and opt for a a less precise much faster version. the faster iterations (resulting in more efficeint training) often trump the gains from a more precise loss. i thikn that might be an idea that is worth keeping in the back of your mind.

these pretty basic changes that took roughly an afternoon to profile, identify, patch and test recduced runtime my a factor of roughly 3x. there are some other optimziations i tried but they go more into the details of what the code is actaully doing and that probably not worth disucssing here. i think this makes the case for doing optimizations not after you are done with reaserach but actually think of optimizations as part of your ml reasaerch. first of all this roughly brought down the cost of an experiment and the time it takes to run an experiment by a factor of 3x. that might speed you up quite significanlty. also  it think modern ml systems need to me designed with hardware in mind. chances are that the most beatiful model/loss will get outperformed by a less expressive loss that omits some of the nuance but is much faster to compute. 

## next steps
for those of you thinking that we are done here, there is more to come. you would be suprised what an karpathy style autoresarech [system](https://deepwiki.com/karpathy/autoresearch) can squeeze out of this program performance wise. but more on that in a future post. 

## installation
make sure that you have nsys installed both on the machine that does the actual work (a remote gcp instance in my case) and the local machine that has a gui where you try to understnad what is happening (my laptop in my case). obviously the two versions you installed need to be compatible. your local device doesn't need nvidia drivers (i.e. works on laptops without nvidia gpus). you can find the download [here](https://developer.nvidia.com/nsight-systems/get-started). for installation/setup instructions as claude, it knows much better than i do.

```
Hint: if you are running nsight systems for the first time it might flashbang you quite badly. you can change it to dark mode in Tools>options>Color Theme
```
<img src="/images/intro-nsys/settings.png" alt="Nsight Systems' Options Menu" width="500"/>  

references:
https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0 [torch compile manual]
https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/ [useful blog]
https://paulbridger.com/posts/nsight-systems-systematic-optimization/ [useful blog]
https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html [cuda programming manual]

### more
(shamelessly plug that we are hiring)
if you find working on things like this interesting, consider reaching out, we are actively hiring.