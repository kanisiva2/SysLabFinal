Image-Guided 3D Reconstruction and Natural Language Editing

Traditional 3D modeling methods often demand specialized software expertise and are time-consuming, posing significant barriers to entry for beginners and hindering efficient replication or modification of real-world objects. This project addresses these challenges by developing an efficient system for 3D model creation and editing from input images, leveraging AI/photogrammetry and natural language processing. The system focuses on three primary use cases: interior design, object case creation, and topographical modeling. A key novelty of our project is the integration of diverse post-creation editing capabilities. For interior design, users can edit models through simple text commands (e.g., "extend width by 5cm"). For object case creation, the system automates the generation of custom-fitted protective cases around imported 3D models with user-defined parameters. In topographical modeling, the project distinguishes itself through novel annotation techniques, including engraving textual information, representing linear features as physical indentations, and integrating dynamic markers. Our solution utilizes the Hunyuan3D 2.0 API for interior design models, Apple's Object Capture API for object case creation, and the OpenTopo API for topographical maps. Text-based editing is implemented using regex for natural language processing, combined with mathematical scaling and manipulation of mesh vertices. This approach aims to make 3D modeling more accessible, flexible, and efficient, demonstrating the ability to generate and precisely edit models of real-world objects. For instance, a 3D model of a couch was successfully created from an image in approximately 3 minutes, with the subsequent 3D print taking about 24 minutes, exhibiting good quality and detail.

Included in this repo:
hydgen - Folder containing Hunyuan3d-2.0 pre-trained model
requirements.txt - all dependencies necessary to run this project
edit.py - Script for interior designing edit functionality
finalapp.py - Code for final interior design website, includes gradio app

In order to run project, simply call finalapp.py
