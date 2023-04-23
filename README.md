Download Link: https://assignmentchef.com/product/solved-10418-hw1-acoustic-speech-recog
<br>
<h1>1          Written Questions</h1>

Answer the following questions in the template provided. Then upload your solutions to Gradescope. You may use L<sup>A</sup>T<sub>E</sub>X or print the template and hand-write your answers then scan it in. Failure to use the template may result in a penalty.

1.1        Reductions to Binary Classification

<ol>

 <li>(1 point) Can we construct an ECOC matrix that yields the same algorithm as One vs. All classification? <em>If yes, describe the matrix; if not, explain why not.</em></li>

 <li>(1 point) Can we construct an ECOC matrix that yields the same algorithm as All vs. All classification?</li>

</ol>

<em>If yes, describe the matrix; if not, explain why not.</em>

<ol start="3">

 <li>(1 point) True or False: One-against-some classifier has logarithmic runtime in the number of examples.</li>

</ol>

True

False

<ol start="4">

 <li>(1 point) True or False: Extreme classification assumes a large label set.</li>

</ol>

True

False

<ol start="5">

 <li>(2 points) For a full binary classification tree of depth <em>d</em>, what is the probability of making a correct classification on <em><sup>X~ </sup></em>. Assume that the probability of a binary classifier at any node making an error is . (A full binary tree is one in which every node, except the leaves, has two children.)</li>

 <li>Numerical answer: For <em>k</em>-way multi-classification with <em>k </em>= 16, how many classifiers need to be constructed for…

  <ul>

   <li>(1 point) …One vs. All classification?</li>

   <li>(1 point) …All vs. All classification?</li>

  </ul></li>

 <li>(1 point) For an arbitrary classification tree with <em>k </em>classes, what is the minimum number of binary classifiers that could be needed?</li>

 <li>(1 point) For an arbitrary classification tree with <em>k </em>classes, What is the maximum number of classifiers that could be needed? (Assume that the sets of classes for each pair of siblings are mutually exclusive.)</li>

</ol>

<h2>1.2        Learning to Search</h2>

In this section, we’ll consider a simple version of learning to search for the task of identifying names in a text. In this setting our action space will be the set {+<em>,</em>−} where + denotes labeling a word as part of a name and − denotes labeling a word as not being part of a name. Our state space will be represented by <em>S<sub>z </sub></em>where <em>z </em>is a vector denoting the labeling of our sentence so far. For example, if we are in state <em>S</em><sub>++− </sub>it means we have labeled word 0 as +, word 1 as +, word 2 as −, and we are currently attempting to label word 3.

Throughout this section, we will be referring exclusively to the following input sentence <em>~x </em>and oracle labeling <em>~y</em>:

In Figure 1.1 you can see a small part of the search space for this problem.

Figure 1.1

<ol>

 <li>(1 point) How many nodes are in the search space for sentence <em>~x</em>?</li>

 <li>Suppose our loss function is Hamming loss.

  <ul>

   <li>(1 point) What does the oracle policy return for <em>S</em><sub>++−</sub>?</li>

   <li>(1 point) What does the oracle policy return for <em>S</em><sub>−−−</sub>?</li>

  </ul></li>

 <li>Suppose our loss function is Hamming loss with an additional error for including only part of a name,</li>

</ol>

e.g. omitting a person’s title. More precisely,

<ul>

 <li>(1 point) What does the oracle policy return for <em>S</em><sub>−</sub>?</li>

 <li>(1 point) What does the oracle policy return for <em>S</em><sub>+</sub>?</li>

</ul>

<ol start="5">

 <li>Suppose that our scoring function is of the form <em>θ</em><em><sup>T </sup></em><strong>f</strong>(<em>x<sub>i</sub>,y<sub>i</sub></em>) where</li>

</ol>

<strong>f</strong>(<em>x<sub>i</sub>,y<sub>i</sub></em>) =[I{<em>x<sub>i </sub></em>starts with a capital letter and <em>y<sub>i </sub></em>= +} I{<em>x<sub>i </sub></em>starts with a capital letter and <em>y<sub>i </sub></em>= −}<em>, </em>I{ is in our gazetteer list and <em>y<sub>i </sub></em>= +}<em>,</em>

I{ is in our gazetteer list and <em>y<sub>i </sub></em>= −}<em>, </em>I{<em>y<sub>i</sub></em><sub>−1 </sub>= −<em>,y<sub>i </sub></em>= −}]

and <em>θ </em>is a vector of length 5. A gazetteer list is essentially a lookup table of entities that we know are names. Our gazetteer list will include { Xavier, Mellon }.

Assume our initial weight vector <em>θ </em>= [−1<em>,</em>1<em>,</em>−1<em>,</em>1<em>,</em>0].

<ul>

 <li>(1 point) What score would our scoring function assign to the ground truth labeling?</li>

 <li>(1 point) What labeling would the greedy policy induced by this scoring function return? Break any ties with +.</li>

 <li>(1 point) Suppose we use the Supervised Approach to Imitation Learning. Which (state, action) pairs would be produced by the learning algorithm? Denote the action by either + or −. Denote a state by the partial sequence it corresponds to (e.g. the state +− corresponds to a state that took action + followed by action −).</li>

 <li>(1 point) Suppose we use DAgger, Which (state, action) pairs would be produced by the learning algorithm? Use the same denotation of states and actions as in the previous question.</li>

</ul>

We decide to train our linear model using the Perceptron update rule. That is, if the classifier (aka. greedy policy) makes a positive mistake (i.e. a mistake where <em>y<sub>i </sub></em>= +) in its action selection, then we <em>add </em>the feature vector to the weights. If it makes a negative mistake (i.e. a mistake where <em>y<sub>i </sub></em>= −), then we <em>subtract </em>the feature vector from the weights. We treat the arrival of each (state, action) pair as a separate online example for the classifier.

<ul>

 <li>(1 point) Using the (state, action) pairs produced by the Supervised Approach to Imitation Learning, what would the new value of <em>θ </em>be after completing the corresponding Perceptron updates?</li>

 <li>(1 point) Using the (state, action) pairs produced by DAgger, what would the new value of <em>θ </em>be after completing the corresponding Perceptron updates?</li>

</ul>

<h2>1.3        Recurrent Neural Network Language Models</h2>

In this section, we wish to use an Elman Network as a building block to design an RNN language model. Below we define the Elman network recursively.

<strong>b</strong><em><sub>t </sub></em>= relu(<strong>B</strong><em><sup>T </sup></em><strong>b</strong><em><sub>t</sub></em>−<sub>1 </sub>+ <strong>A</strong><em><sup>T </sup></em><strong>a</strong><em><sub>t </sub></em>+ <strong>d</strong>) <strong>c</strong><em><sub>t </sub></em>= softmax(<strong>C</strong><em><sup>T </sup></em><strong>b</strong><em><sub>t </sub></em>+ <strong>e</strong>)

where <strong>a</strong><em><sub>t</sub></em><em>,</em>∀<em>t </em>is given as input to the network; <strong>A</strong><em>,</em><strong>B</strong><em>,</em><strong>C</strong><em>,</em><strong>d</strong><em>,</em><strong>e </strong>are parameters of the network; and the initial hidden units <strong>b</strong><sub>0 </sub>are also treated as parameters. In this problem, we assume <strong>a</strong><em><sub>t</sub></em><em>,</em><strong>b</strong><em><sub>t</sub></em><em>,</em><strong>c</strong><em><sub>t </sub></em>∈ R<sup>2 </sup>for all <em>t</em>, i.e. all vectors have length two, and that the parameters matrices and vectors are appropriately defined to preserve those dimensions. Above, the function relu(·) is the Rectified Linear Unit function applied elementwise to its vector-valued parameter. The function softmax(·) is likewise vector-valued and defined below. We use [·]<em><sub>i </sub></em>to explicitly denote the <em>i</em>th element of its argument.

[relu(<strong>v</strong>)]<em><sub>i </sub></em>, max(0<em>,v<sub>i</sub></em>)

[softmax

Figure 1.2 depicts the Elman Network. The parameters are shown in red, but their connections into the computation graph are not shown.

Figure 1.2

Assume that we wish to build a language model <em>p</em>(<strong>y</strong>) of binary sequences <strong>y </strong>∈ {+<em>,</em>−}<em><sup>L </sup></em>of fixed length <em>L</em>. We have training data consisting of sequences D = {<strong>y</strong><sup>(1)</sup><em>,…,</em><strong>y</strong><sup>(<em>N</em>)</sup>}, , where |<strong>y</strong><sup>(<em>i</em>)</sup>| = <em>L</em>. Assume further that we have pre-encoded the data as sequences of one-hot vectors D<sup>0 </sup>= {<strong>z</strong><sup>(1)</sup><em>,…,</em><strong>z</strong><sup>(<em>N</em>)</sup>} such that:

if, then <strong>z</strong>, if, then <strong>z</strong>.

For such a pair of vectors, we write that <strong>z</strong><sup>(<em>i</em>) </sup>= one-hot(<strong>y</strong><sup>(<em>i</em>)</sup>)

<ol>

 <li>(1 point) Short answer: Since we wish to treat this Elman Network as an RNN-LM, how many inputs <strong>a</strong><em><sub>t </sub></em>will we need for a single training example <strong>y </strong>∈ D with |<strong>y</strong>| = <em>L</em>? Explain your answer.</li>

 <li>(1 point) Select one: If we decide to train this RNN-LM with <em>Teacher Forcing</em>, we will need to compute a loss function for an input example <strong>y </strong>∈ D. Assuming so, how would we define <strong>a</strong><em><sub>t</sub></em>? <em>Note: Be sure to account for all </em><em>t in your definition.</em></li>

 <li>(1 point) Select one: If we decide to train this RNN-LM with <em>Scheduled Sampling</em>, we will need to compute a loss function for an input example <strong>y </strong>∈ D. Assuming we use a schedule that always selects the model policy with probability 1, how would we define <strong>a</strong><em><sub>t</sub></em>? <em>Note: Be sure to account for all </em><em>t in your definition.</em></li>

 <li>(1 point) Write the cross entropy loss <em>` </em>for a single training example <strong>z </strong>∈ D<sup>0 </sup>in terms of the units and/or parameters of the RNN-LM: <strong>a</strong><em><sub>t</sub></em><em>,</em><strong>b</strong><em><sub>t</sub></em><em>,</em><strong>c</strong><em><sub>t</sub></em><em>,</em><strong>A</strong><em>,</em><strong>B</strong><em>,</em><strong>C</strong><em>,</em><strong>d</strong><em>,</em><strong>e</strong>.</li>

</ol>

Suppose we have parameter values as defined below:

<h3><strong>A </strong><strong>         B </strong><strong>  C </strong>          (1.1) <strong>d </strong><strong>      e </strong><strong> b</strong>              (1.2)</h3>

<ol start="5">

 <li>Numerical answer: When computing the probability of the sequence <strong>y </strong>= [+<em>,</em>−<em>,</em>+], what is the value of the following three quantities? <em>Note: Round each numerical value to two significant figures.</em>

  <ul>

   <li>(1 point) <em>b</em><sub>1<em>,</em>1 </sub>=</li>

   <li>(1 point) <em>b</em><sub>2<em>,</em>1 </sub>=</li>

  </ul></li>

 <li>Numerical answer: When computing the probability of the sequence <strong>y </strong>= [+<em>,</em>−<em>,</em>+], what is the value of the following three quantities? <em>Note: Round each numerical value to two significant figures.</em>

  <ul>

   <li>(1 point) <em>c</em><sub>1<em>,</em>1 </sub>=</li>

   <li>(1 point) <em>c</em><sub>2<em>,</em>1 </sub>=</li>

  </ul></li>

 <li>(1 point) Numerical answer: What is the probability of the sequence <strong>y </strong>= [+<em>,</em>−<em>,</em>+] according to this RNNLM? <em>Note: Round the numerical value to two significant figures.</em></li>

</ol>

<em>p</em>(<strong>y</strong>) =

<ol start="8">

 <li>(1 point) Numerical answer: What is the probability of the <em>(length one!) </em>sequence <strong>y</strong><sup>0 </sup>= [−] according to this RNNLM? <em>Note: Round the numerical value to two significant figures.</em></li>

</ol>

<em>p</em>(<strong>y</strong><sup>0</sup>) =

<h2>1.4        Empirical Questions</h2>

The following questions should be completed after you work through the programming portion of this assignment (Section 2).

<ol>

 <li>(10 points) Record your model’s performance on the test set after 10 epochs in terms of Cross Entropy (CE) and Character Error Rate (CER) when trained with the following schemes. <em>Note: Round each numerical value to two significant figures.</em></li>

</ol>

<table width="236">

 <tbody>

  <tr>

   <td width="128">Schedule</td>

   <td width="54">CE</td>

   <td width="54">CER</td>

  </tr>

  <tr>

   <td width="128">All Oracle</td>

   <td width="54"> </td>

   <td width="54"> </td>

  </tr>

  <tr>

   <td width="128">All Model</td>

   <td width="54"> </td>

   <td width="54"> </td>

  </tr>

  <tr>

   <td width="128"><em>β </em>= 0<em>.</em>75</td>

   <td width="54"> </td>

   <td width="54"> </td>

  </tr>

  <tr>

   <td width="128">Linear Decay</td>

   <td width="54"> </td>

   <td width="54"> </td>

  </tr>

  <tr>

   <td width="128">Exponential Decay</td>

   <td width="54"> </td>

   <td width="54"> </td>

  </tr>

 </tbody>

</table>

<ol start="2">

 <li>(10 points) Plot training and testing cross entropy curves for three different training procedures: <em>All Oracle</em>, <em>All Model</em>, and the fixed <em>β </em>= 0<em>.</em>75 training procedure. Let the <em>x</em>-axis ranges over 10 epochs.</li>

</ol>

<em>Note: Your plot must be machine generated.</em>

<ol start="3">

 <li>In class we saw that we can prove a no-regret bound for sequences of <em>β </em>such that</li>

</ol>

.

<ul>

 <li>(1 point) Show that a fixed <em>β </em>does not satisfy this condition.</li>

 <li>(1 point) Show that exponential decay does satisfy this condition.</li>

 <li>(1 point) Did this theoretical difference make a difference in practice? Briefly explain why or why not with respect to this dataset and problem setting.</li>

</ul>

<h2>1.5        Wrap-up Questions</h2>

<ol>

 <li>(1 point) Multiple Choice: Did you correctly submit your code to Autolab?</li>

</ol>

Yes

No

<ol start="2">

 <li>(1 point) Numerical answer: How many hours did you spend on this assignment?.</li>

</ol>

<h1>2          Programming</h1>

Your goal in this assignment is to implement a deep learning model for acoustic speech recognition (ASR). You will implement a function to encode speech data into a fixed length vector, a function to decode this fixed length vector into a sequence of characters, and a variety of strategies to train the overall model.

Your solution for this section must be implemented in PyTorch using the data files we have provided to you. This restriction is because we will be grading your code by hand to check your understanding as well as your model’s performance.

<h2>2.1        Task Background</h2>

Acoustic speech recognition is the problem of taking in recordings of people talking and predicting what was said. While many successful models try to predict linguistically meaningfully units like phonemes, we will be directly predicting characters from waveforms for simplicity.

Though we will train our model with a standard classification loss (Cross-Entropy), what we really care about is the accuracy of our transcription. For a given target sentence, we define the Character Error Rate as the number of character deletions (<em>d</em>), character insertions (<em>i</em>), and character substitutions (<em>s</em>) need to transform the output transcription to the goal transcription over the number of characters in the output transcription (n).

<h3>                                                                                       CER                                                                             (2.1)</h3>

Conveniently, this equation can be calculated with the following snippet of code, which runs a dynamic programming algorithm to compute edit distance:

<strong>from </strong>nltk.metrics.distance <strong>import </strong>edit_distance

cer = edit_distance(our_string, goal_string) / <strong>len</strong>(our_string)

<h2>2.2        Data</h2>

In order to reduce your workload and computational requirements, we have preprocessed a subset of the TIMIT dataset for you to use in this task.

The data is divided into two folders, train and test. In each folder is a sequence of pairs of numpy files (.npy) and text files (.txt). If a numpy file is named ”xyz.npy”, it’s corresponding transcription can be found in ”xyz.txt”.

Given Input: Preprocessed audio files in numpy array of size 20 × <em>L</em>, where <em>L </em>is the number of timesteps in the audio file.

Goal Output: Preprocessed text files of the transcribed audio.

For this section’s points, you will need to implement data loaders for PyTorch so that we can efficiently train our model batch by batch. Note that because we are working with sequences, we will need all sequences in a batch to have the same length.

(Hint: PyTorch allows you to pass a ”collate fn” to torch.utils.data.DataLoader and provides a function called ”pad sequence” in torch.nn.utils.rnn)

<h2>2.3        Seq2Seq Implementation</h2>

A sequence to sequence model (commonly abbreviated Seq2Seq) is a model that takes in a sequence of input, like words in an English sentence or samples of a waveform, and outputs another sequence, like words in Arabic or transcribed phonemes. Models of this type are frequently used in machine translation, text to speech applications, and automatic speech recognition.

Seq2Seq models comprise of two parts, an encoder and a decoder. The encoder takes in a sequence of inputs and outputs an <em>encoding </em>that represents all of the information contained in the input. The decode then takes in this encoding and outputs a sequence in the new domain. This process can be seen in the figure below.

Figure 2.1: The encoder and decoder are trained jointly in a Seq2Seq model.

For this section, you must implement a working Seq2Seq model. Your implementation must have both the encoder and decoder as single-layer LSTMs with 50% dropout applied to the input layer to the encoder. Every hidden dimension in your neural networks should be 128 and your embedding size should be 256. Set your optimizer to be Adam with the default PyTorch parameters. Your program should be able to run quickly and easily on a laptop due to the simplicity of our model and the limited size of our data.

<h2>2.4        DAgger Implementation</h2>

As we’ve seen in class, DAgger is an algorithm for collecting training examples for our model by sometimes following the actions performed by an expert and sometimes following our model’s decisions.

For this section’s points, you will need to implement DAgger as described in Algorithm 1. Note that this algorithm allows <em>β<sub>i </sub></em>to vary with timestep <em>i</em>. This allows us to explore various schedules for the <em>β </em>parameter, including some that don’t have the theoretical guarantees discussed in class.

Please implement a general version of DAgger and then the code necessary to run DAgger with (1) a fixed <em>β</em>, (2) a linearly decaying <em>β </em>where <em>β </em>is decreased by 0<em>.</em>05 after each epoch, and (3) an exponentially decaying <em>β</em>, where <em>β </em>= exp−1 × <em>i </em>where <em>i </em>is the current epoch.

Algorithm 1 DAgger

Initialize D ← ∅.

Initialize <em>π</em>ˆ<sub>1 </sub>to any policy in Π. for i=1 to N do

Let <em>π<sub>i </sub></em>= <em>β<sub>i</sub>π</em><sup>∗ </sup>+ (1 − <em>β<sub>i</sub></em>)<em>π</em>ˆ<em><sub>i</sub></em>

Sample <em>T</em>-step trajectories using <em>π<sub>i</sub></em>.

Get dataset D<em><sub>i </sub></em>= {<em>s,π</em><sup>∗</sup>(<em>s</em>)} of visited states by <em>π<sub>i </sub></em>and actions given by the expert.

Aggregate datasets: D ← D ∪ D<em><sub>i</sub></em>.

Train classifier <em>π</em>ˆ<em><sub>i</sub></em><sub>+1 </sub>on D.

end for return best <em>π</em>ˆ<em><sub>i </sub></em>on validation.

<h2>2.5        Autolab Submission</h2>

You must submit a .tar file named seq2seq.tar containing seq2seq.py, which contains all of your code.

You can create that file by running:

tar -cvf seq2seq.tar seq2seq.py

from the directory containing your code.

Some additional tips: DO NOT compress your files; you are just creating a tarball. Do not use tar -czvf. DO NOT put the above files in a folder and then tar the folder. Autolab is case sensitive, so observe that all your files should be named in lowercase. You must submit this file to the corresponding homework link on Autolab.

Your code will not be autograded on Autolab. Instead, we will grade your code by hand; that is, we will read through your code in order to grade it. As such, please carefully identify major sections of the code via comments.

<ol>

 <li></li>

</ol>