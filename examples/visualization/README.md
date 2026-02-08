# Visualization in Luminal

## Design Choices
Luminal produces intermediate files rather than complete visualizations

The two primary file types are: 
- `.html` files
- `.dot` files

These files enable interactive viewing which is often necessary for making visualizations interpretable. 

## VSCode Extensions
We recommend the following extensions for VSCode users. 
The integrated nature of these extensions makes viewing these files easy even on remote machines via ssh. 

- `Live Preview` by microsoft. 
- `Graphviz Interactive Preview` by tintinweb 

## Example Provided 
In the example program, as simple program is defined. 
From this a HLIR graph is created and visualized. 
A saturated EGraph is created and visualized. 
Finally an LLIR graph is extracted and visualized. 
