# Intelligent-image-search-engine

Functionalities reflecting the idea of an intelligent image search system based on computer vision and image processing methods have been implemented. Here are the functionalities that have been implemented: 
* Search for similar images 
  * by color histogram method
  * using the SIFT method
  * Using a pre-trained medele (INCEPTION) as a feature extractor
* Locating objects in an image folder from YOLO 
* Facial recognition in an image folder using combined methods ( landmark prediction , SIFT )
* Brand-targeted shoe recommendation
* Image search from image fragments (with fragment localization)
* Search for images containing specific text

**NB** : The idea was not to necessarily use the most effective methods for each feature, but to explore different methods and appreciate their capabilities.

You'll find :
* Intelligent image search engine.ipynb : The main notebook containing details of our work
* Notebook Interface.ipynb : A notebook with a small integrated interface for easy handling of the various functions
 *  <a target="_blank" rel="noopener noreferrer" href="/interface_notebook_output.png"><img width="100%" src="/interface_notebook_output.png" style="max-width: 100%;"></a>
* "tkinter app" folder containing the code for our tkinter application, to provide a better desktop interface for manipulating our functions
 *  <video width="560" height="315" controls><source src="tkapp_video.mp4" type="video/mp4"></video>

If you need the drive containing the weights of our models, as well as other files to make these notebooks work properly, please don't hesitate to contact us,
