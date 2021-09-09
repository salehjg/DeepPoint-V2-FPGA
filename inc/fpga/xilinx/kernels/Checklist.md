# Kernel Wrapper Integrity Checklist
* Make sure to use crossed-bank input tensors' shapes in the algorithms.
* The book-keeping list includes raw inputs, bank-crossed inputs, outputs
* The dependency vector (events wait list) includes bank-crossed inputs
* In case of expanding inputs to the target rank, make sure that:
    - The bank-crossed version is being expanded `diff` times.
    - The same bank-crossed version is squeezed `diff` times.
    - Depending on the kernel, the output tensor is squeezed `diff` times.
* The enqueue member function returns the output tensor.
* Make sure that the final output tensor's event is set by the last enqueueTask.
* Make sure that the tensor tag is set to the final output tensor of the enqueue kernel launch.
* Make sure that the tensor tags are read from the actual inputs and not their cross-banked versions.
