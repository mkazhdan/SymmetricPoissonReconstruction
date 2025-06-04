<center><h2>Symmetric Poisson Reconstruction (Version 1.00)</h2></center>
<center>
<a href="http://www.cs.jhu.edu/~misha/Code/SymmetricPoissonRecon/Version1.00/index.html#LINKS">links</a>
<a href="http://www.cs.jhu.edu/~misha/Code/SymmetricPoissonRecon/Version1.00/index.html#EXECUTABLES">executables</a>
<a href="http://www.cs.jhu.edu/~misha/Code/SymmetricPoissonRecon/Version1.00/index.html#USAGE">usage</a>
<a href="http://www.cs.jhu.edu/~misha/Code/SymmetricPoissonRecon/Version1.00/index.html#CHANGES">changes</a>
</center>
<hr>
This software supports reconstruction of surfaces from unoriented points.
<hr>
<a name="LINKS"><b>LINKS</b></a><br>
<ul>
<b>Papers:</b>
<a href="http://www.cs.jhu.edu/~misha/MyPapers/SGP25.pdf">[Kohlbrenner, Liu, Alexa, and Kazhdan, 2025]</a>,
<br>
<b>Executables: </b>
<a href="http://www.cs.jhu.edu/~misha/Code/SymmetricPoissonRecon/Version1.00/SymPR.x64.zip">Win64</a><br>
<b>Source Code:</b>
<a href="http://www.cs.jhu.edu/~misha/Code/SymmetricPoissonRecon/Version1.00/SymPR.zip">ZIP</a> <a href="https://github.com/mkazhdan/SymmetricPoissonRecon">GitHub</a><br>
<!--
<b>Older Versions:</b>
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version9.011/">V9.011</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version9.01/">V9.01</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version9.0/">V9.0</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version8.0/">V8.0</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version7.0/">V7.0</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version6.13a/">V6.13a</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version6.13/">V6.13</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version6.12/">V6.12</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version6.11/">V6.11</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version6.1/">V6.1</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version6/">V6</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version5.71/">V5.71</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version5.6/">V5.6</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version5.5a/">V5.5a</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version5.1/">V5.1</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version5/">V5</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version4.51/">V4.51</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version4.5/">V4.5</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version4/">V4</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version3/">V3</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version2/">V2</a>,
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version1/">V1</a>
-->
</ul>
<hr>
<a name="EXECUTABLES"><b>EXECUTABLES</b></a><br>


<ul>
<dl>
<DETAILS>
<SUMMARY>
<font size="+1"><b>SymmetricPoissonRecon</b></font>:
Reconstructs a polygon/triangle mesh from a set of unoriented 3D points by using the point-set to define a symmetric matrix field and solving the quartic optimization problem for the implicit function such that the outer-produce of the function's gradient with itself best matches the matrix field.
</SUMMARY>
<dt><b>--in</b> &lt;<i>input points</i>&gt;
</dt><dd> This string is the name of the file from which the point set will be read.<br>
If the file extension is <i>.ply</i>, the file should be in
<a href="http://www.cc.gatech.edu/projects/large_models/ply.html">PLY</a> format, giving the list of oriented
vertices with the x-, y-, and z-coordinates of the positions encoded by the properties <i>x</i>, <i>y</i>, and
<i>z</i>.<br>
Otherwise, the file should be an ascii file with groups of 3, white space delimited, numbers: the x-, y-, and z-coordinates of the point's position. (No information about the number of oriented point samples should be specified.)<br> 

</dd><dt>[<b>--out</b> &lt;<i>output mesh/implicit-function</i>&gt;]
</dt><dd> This string is the name of the file to which the output will be written (output in <a href="http://www.cc.gatech.edu/projects/large_models/ply.html">PLY</a> format).<br>
<UL>
<LI>If the file extension is <i>.ply</i>, the output will be a polygon/triangle mesh.<br>
<LI>If the file extesnion is <i>.grid</i>, the output will be regular grid sampling the implicit function. The file will consist of seven header lines in ASCII describing the contents, following by the grid values in binary.<BR>
<UL>
<LI>The header looks like:<PRE><CODE>G3
1 DOUBLE
&lt;RES_X&gt; &lt;RES_Y&gt; &lt;RES_Z&gt;
&lt;M_00&gt; &lt;M_01&gt; &lt;M_02&gt; &lt;M_03&gt;
&lt;M_10&gt; &lt;M_11&gt; &lt;M_12&gt; &lt;M_13&gt;
&lt;M_20&gt; &lt;M_21&gt; &lt;M_22&gt; &lt;M_23&gt;
&lt;M_30&gt; &lt;M_31&gt; &lt;M_22&gt; &lt;M_33&gt;
</CODE></PRE>
The first two lines describe the contents of the file -- a 3D grid with a double-precision floating point value per cell.<br>
The next line gives the resolution of the grid in <code>x</code>-, <code>y</code>-, and <code>z</code>-directions.<br>
The following four lines give the 4x4 coefficients of the homogenous transformation <CODE>&lt;M&gt;</CODE> taking grid-space coordinates to world-coordinates.
<LI>
The next 8 x <code>&lt;RES_X&gt;</code> x <code>&lt;RES_Y&gt;</code> x <code>&lt;RES_Z&gt;</code> bytes correspond to the (double-precision) floating point values
of the implicit function.
</UL>

</dd><dt>[<b>--depth</b> &lt;<i>reconstruction depth</i>&gt;]
</dt><dd> This integer is the depth of the tree that will be used for surface reconstruction.
Running at depth <i>d</i> corresponds to solving on a voxel grid whose resolution is 
2^<i>d</i> x 2^<i>d</i> x 2^<i>d</i>. Note that since the reconstructor adapts the octree to the
sampling density.<BR>
The default value for this parameter is 7.

</dd><dt>[<b>--sWeight</b> &lt;<i>screening weight</i>&gt;]
</dt><dd> This floating point value specifies the weight given to having the implicit function evaluate to zero at the samples.<br>
The default value is 50,000.

</dd><dt>[<b>--dWeight</b> &lt;<i>boundary Dirichlet weight</i>&gt;]
</dt><dd> This floating point value specifies the weight given to having the implicit function be constant on the boundary of the unit-cube.<br>
The default value is 100.

</dd><dt>[<b>--scale</b> &lt;<i>scale factor</i>&gt;]
</dt><dd> This floating point value specifies the ratio between the diameter of the cube used for reconstruction
and the diameter of the samples' bounding cube.<br>
The default value is 1.25.

</dd><dt>[<b>--coarseIiters</b> &lt;<i>Gauss-Seidel iterations at coarsest level</i>&gt;]
</dt><dd> This integer value specifies the number of Gauss-Seidel relaxations to be performed at the coarsest level of the hierarchy.<br>
The default value for this parameter is 512.

</dd><dt>[<b>--iters</b> &lt;<i>Gauss-Seidel iterations at finest level level</i>&gt;]
</dt><dd> This integer value specifies the number of Gauss-Seidel relaxations to be performed at the finest level of the hierarchy.<br>
The default value for this parameter is 10.

</dd><dt>[<b>--iMult</b> &lt;<i>Gauss-Seidel iteration multiplier</i>&gt;]
</dt><dd> This floating point value how the multiplicative factor for the number of Gauss Seidel iterations to be performed at progressively coarser levels (excluding the coarsest level).<BR>
The default value for this parameter is 2.0.

</dd><dt>[<b>--nn</b> &lt;<i>number of nearest neighbors</i>&gt;]
</dt><dd> This integer value specifies the number of nearest neighobrs that are to be used for estimating the local covariance.<BR>
The default value for this parameter is 20.

</dd><dt>[<b>--density</b>]
</dt><dd> Enabling this flag tells the reconstructor to output a triangle mesh (rather than a polygon mesh).

</dd><dt>[<b>--verbose &lt;<i>verbosity</i>&gt;</b>]
</dt><dd> This integer variable specifies the level of verbosity output to the standard output (with larger values corresponding to more verbose output).<BR>
<B>Note</B>: The calculation of verbose output can noticeably affect run-time performance.

</dd>
</DETAILS>
</dl>
</ul>



<hr>
<a name="USAGE"><b>USAGE EXAMPLES (WITH SAMPLE DATA)</b></a><br>

<ul>
<dl>
<DETAILS>
<SUMMARY>
<font size="+1"><b>SymmetricPoissonRecon </b></font>
</SUMMARY>
For testing purposes, a point-set sampling the <A HREF="https://www.cs.jhu.edu/~misha/SymmetricPoissonReconstruction/bunny_raw.xyz">Stanford Bunny</A> is provided. (Data sourced from the <A HREF="http://graphics.stanford.edu/data/3Dscanrep/">Stanford Scanning Repository</A>.) The point-set consists of roughly 360K points, written in ASCII format with three floating point values per line.<BR>

The surface of the model can be reconstructed by calling symmetric Poisson reconstructor:
<blockquote><code>% SymmetricPoissonRecon --in bunny_raw.xyz --out bunny.ply</code></blockquote>

</DETAILS>
</dl>
</ul>



<hr>
<DETAILS>
<SUMMARY>
<font size="+1"><b><B>HISTORY OF CHANGES</B></b></font>
</SUMMARY>
<a href="http://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version1.00/">Version 1</a>:
<ul>
<li> Initial source-code.
</Ul>
</DETAILS>
