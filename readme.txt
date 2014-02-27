Bayesian Matting Demo

Author: Yili Zhao (panovr at gmail dot com)

1. Introduce

This demo implements Yung-Yu Chuang's "A Bayesian Approach to Digital Matting" paper [1].

This demo consists of two parts:

(1) xxx-matting.cmd will start the console application;

(2) Bayesian_Matting_GUI will start the window application with a nice GUI.

2. Console application usage

(1) Double-click "gandalf-matting.cmd" or "knockout-matting.cmd" 
or "lighthouse-matting.cmd";

(2) After "src", "trimap" and "composite" three windows displayed, click "enter" key;

(3) Then the solving process starts, and after several minutes, the solved alpha map
will be displayed in "alphamap" window;

(4) Click "enter" key again, the composited image with new background will be displayed
in "result' window;

(5) Click "enter" key to quit the application.

3. Window application usage

(1) Select "Load source..." from "File" menu and open the source image;

(2) Select "Load trimap..." from "File" menu and open the trimap image;

(3) Select "Create alpha" from "Render" menu and wait for some minutes;

(4) After alphamap displayed, select "Load background..." from "Composite" menu and
select the new background image;

(5) Then the new composited image will be displayed.
 
Reference
[1] Yung-Yu Chuang, Brian Curless, David H. Salesin, and Richard Szeliski. A Bayesian
Approach to Digital Matting. In Proceedings of IEEE Computer Vision and Pattern 
Recognition (CVPR 2001), Vol. II, 264-271, December 2001.