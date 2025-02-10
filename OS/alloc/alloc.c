#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// make clean all; ./test -v free-reuse

//if you dont have space, keep doubling until you do, do this by having initial space
// check if theres space by summing all size and comparing to current max size

typedef struct obj_metadata {        //this is per chunk? implement in free, malloc
 size_t size;                   // in malloc: init the struct of 1 chunk
 struct obj_metadata *next;     // in realloc: only change size
 struct obj_metadata *prev;     //in calloc: don't do anything since its handled in malloc? might be issue with multiple and only 1 pointer to prev and next
 int is_free;
} obj_metadata;

obj_metadata *head = NULL;  //start of memory

obj_metadata *freelist = NULL; //only add pointers to this when freed

void add_block_to_heap(size_t size, void *ptr){

    obj_metadata *new_block = ptr;
    new_block->next = NULL;
    new_block->prev = head;
    new_block->is_free = 0;      //not free
    new_block->size = size;

    head = new_block;
}

void add_to_freelist(size_t size, void *ptr){

    if(freelist == NULL){
        freelist = ptr;
        freelist->next = NULL;
        freelist->prev = NULL;
        freelist->is_free = 1;      //free
        freelist->size = size;
    }

    else{
        obj_metadata *new_block = ptr;
        new_block->next = NULL;
        new_block->prev = freelist;
        new_block->size = size;
        new_block->is_free = 1;
        freelist = new_block;
    }   
}

void *mymalloc(size_t size)
{  
    if(size == 0) return NULL;

    if(size % 8 != 0) {
        while(size % 8 != 0){
            size ++;
        }
    }

    //go through list of blocks, if free -> check if same size -> place new pointer there
    
    
    obj_metadata *tail = head;  
    while (tail != NULL) {
        if (tail->is_free == 1 && tail->size >= size) {
            tail->is_free = 0;
            return ((char*) tail + sizeof(obj_metadata));
        }
        tail = tail->prev;
    } //works but slow

/*
    obj_metadata *tail = freelist;    //check if there's a free block in the freelist, use it
    fprintf(stderr, "Freelist %x,", freelist);
    while (tail != NULL) {
        if (tail->is_free == 1 && tail->size >= size) {
            tail->is_free = 0;
            return ((char*) tail + sizeof(obj_metadata));
        }
        tail = tail->prev;
    }*/   

    void *currlocation = sbrk(sizeof(obj_metadata));  //for metadata

    if(head == NULL){
        head = currlocation;
        head->next = NULL;
        head->prev = NULL;
        head->is_free = 0;      //not free
        head->size = size;
    }  
    else{
        add_block_to_heap(size, currlocation);  
    }
    
    sbrk(size);     //alloc for data needed
    return (char*)currlocation + sizeof(obj_metadata);  //char makes the pointer of currlocation to bytes, to make byte addition possible
                                                    //return location at start of actual data (end of metadata)
}

void *mycalloc(size_t nmemb, size_t size)
{
    if(size == 0 || nmemb == 0){
        return NULL;
    }

    void *currlocation = mymalloc(nmemb * size);
    memset(currlocation, 0, nmemb * size);          //set memory to 0
    
    return currlocation;
}

void myfree(void *ptr)
{
    if(ptr == NULL) return;
    obj_metadata *object = ((obj_metadata*) ptr)-1;
    object->is_free = 1;
    //add_to_freelist(object->size,ptr);
}

void *myrealloc(void *ptr, size_t size){
    if(ptr == NULL){
        return mymalloc(size);
    }

    else if(size == 0 && ptr != NULL){
        myfree(ptr);
        return NULL;
    }
    
    else{
        obj_metadata *object = ((obj_metadata*) ptr)-1;
        if(object->size < size){
            void *newlocation = mymalloc(size);
            memcpy(newlocation,ptr,object->size);  //copy data of old block, to newly allocated block
            myfree(ptr);
            return newlocation;
        }

        else{
            object->size = size;
            return ptr;
        }
    }
}


/*
 * Enable the code below to enable system allocator support for your allocator.
 * Doing so will make debugging much harder (e.g., using printf may result in
 * infinite loops).
 */
#if 0
void *malloc(size_t size) { return mymalloc(size); }
void *calloc(size_t nmemb, size_t size) { return mycalloc(nmemb, size); }
void *realloc(void *ptr, size_t size) { return myrealloc(ptr, size); }
void free(void *ptr) { myfree(ptr); }
#endif
