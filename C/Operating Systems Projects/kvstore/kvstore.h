#ifndef KVSTORE_H
#define KVSTORE_H

#include "common.h"
#include "hash.h"
//user defined variables should be here, variables can be used as synchronization, locks and mutex variables

struct user_item {
    pthread_rwlock_t itemMutex;
    pthread_cond_t condItem;

    // Add your fields here.
    // You can access this structure from ht_item's user field defined in hash.h
};

struct user_ht {
    pthread_rwlock_t* bucket_locks; // Array of bucket mutexes
    pthread_cond_t condBucket;

    // Add your fields here.
    // You can access this structure from the hashtable_t's user field define in has.h
};

#endif
