# Segment Labeling

The code in this folder provides high-level access to segmenting and labeling text.

### Date

Our custom date class is defined in `Date.py`. It contains exactly the date information
we need and provides sorting.

### Labeling

`Labeling.py` abstracts away most of the work. The main function here is `label()`.
Pass a list of strings (ideally containing segments of a text) and it will return
a list of tuples, each containing the string and the assigned `Date` object. The 
implementation used for date extraction can be changed by swapping out the
`xyz_label_segment()` call in `label()`.
