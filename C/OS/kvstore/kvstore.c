#include <semaphore.h>
#include <assert.h>
#include <pthread.h>
#include <stdbool.h>

#include "server_utils.h"
#include "common.h"
#include "request_dispatcher.h"
#include "hash.h"
#include "kvstore.h"

// DO NOT MODIFY THIS.
// ./check.py assumes the hashtable has 256 buckets.
// You should initialize your hashtable with this capacity.
#define HT_CAPACITY 256

#define MAX_THREADS 5 // Number of threads in the pool
#define MAX_QUEUE_SIZE 100

int set_request(int socket, struct request *request)
{   
    //fprintf(stderr, "trynna do SET\n");
    size_t expected_len = request->msg_len;
    pr_info("Read %zu\n", expected_len);
    hash_item_t *item = NULL;
    char *key = request->key;
    int err = 0;
    //race conditions need to be inserted so that no concurrent access to bucket, use lock and unlock for item and bucket

    unsigned int h = hash(key) % ht->capacity;   //get bucket
    hash_item_t *item_in_bucket = ht->items[h];
    hash_item_t *olditem = ht->items[h];


    //get the item 
    while(item_in_bucket != NULL){
        if(!strcmp(item_in_bucket->key,key)){
            item = item_in_bucket;
            break;
        }
        item_in_bucket = item_in_bucket->next;
    }

    if(item == NULL){  //key is not in bucket
        //fprintf(stderr,"item doesn exist\n");
        //lock the bucket, should it be on individual buckets? and should we implement the cond wait?
        //bucketlock = pthread_mutex_trylock(&(ht->user->bucket_locks[h])); 
        pthread_rwlock_trywrlock(&(ht->user->bucket_locks[h])); 
        //if(bucketlock != 0){
            //fprintf(stderr,"bucket couldnt be locked");
        //}

        hash_item_t *new_item = malloc(sizeof(hash_item_t));
        new_item->user = (void *)malloc(sizeof(struct user_item));
        new_item->prev = NULL;
        new_item->next = ht->items[h];

        if(ht->items[h] != NULL){
            ht->items[h]->prev = new_item;
        }
        ht->items[h] = new_item;  //set head of ht to the new item

        new_item->key = malloc(strlen(key) +1);

        strcpy(new_item->key,key);
        item = new_item;      //now the item is found in the bucket after inserting it        
        pthread_rwlock_init(&(item->user->itemMutex),NULL);
        //unlock the bucket
        pthread_rwlock_unlock(&(ht->user->bucket_locks[h]));
    }

    if(pthread_rwlock_trywrlock(&(item->user->itemMutex))!= 0){
        err = 1;  
    }
    //fprintf(stderr,"locking %s gave code %d\n",item->key,lock_result);
    

    size_t total_received = 0;
    
    if(err == 1){                       
        //fprintf(stderr, "dumping\n");
        char* discard = malloc(expected_len);
        //item->value = malloc(expected_len+1);
        while (total_received < expected_len) {
            size_t remaining = expected_len - total_received;
            size_t rcved = read_payload(socket, request, remaining, discard+total_received);
            //fprintf(stderr, "test %d\n", rcved);
            total_received += rcved;
        }

        //fprintf(stderr, "dumped %d\n", total_received);
        //free(discard);
        send_response(socket,KEY_ERROR,total_received, discard);

        //request->connection_close = 1;
        //close_connection(socket);
        return -1;
    }        

    else{
        char rcfbuf[expected_len];  //try using this for abort
        item->value = malloc(expected_len);
        item->value_size = expected_len;

        while (total_received < expected_len) {
            size_t remaining = expected_len - total_received;
            //fprintf(stderr,"remaining %d\n", remaining);
            //fprintf(stderr,"key is %s\n",item->key);
            int rcved = read_payload(socket, request, remaining, rcfbuf);
            //fprintf(stderr,"just read  %d\n", rcved);
            
            if (rcved < 0) {  

                if(olditem != NULL){  //item already existed so just write old value to new
                    //fprintf(stderr,"1");
                    item->value = strdup(olditem->value);
                }
                else{
                    if(item->prev == NULL){

                        if(item->next == NULL){
                            //fprintf(stderr,"2");
                            ht->items[h] = NULL;
                        }

                        else{
                            //fprintf(stderr,"3");
                            item->next->prev = NULL;
                            ht->items[h] = item->next;
                        }
                        
                    }
                    else{
                        if(item->next != NULL){
                            //fprintf(stderr,"4");
                            item->next->prev = item->prev;
                            item->prev->next = item->next;
                        }
                    }
                }

                pthread_rwlock_unlock(&item->user->itemMutex);
                send_response(socket, STORE_ERROR, 0, NULL);            
                //request->connection_close = 1;
                //close_connection(socket);
                return total_received;
                
            }
            else{
                memcpy(item->value+total_received,rcfbuf,rcved);
                total_received += rcved;                
            }

        }
    }
    

    

    check_payload(socket, request, expected_len);
    send_response(socket, OK, expected_len, item->value);
    pthread_rwlock_unlock(&item->user->itemMutex); 
    //fprintf(stderr, "unlock from SET was %d\n", result_un);
    return total_received;
}

int get_request(int socket, struct request *request)
{
    hash_item_t *item = NULL;
    char *key = request->key;
    //fprintf(stderr, "trynna access %s\n", key);

    //race conditions need to be inserted so that no concurrent access to bucket, use lock and unlock for item and bucket
    unsigned int h = hash(key) % ht->capacity;   //get bucket
    
    if(pthread_rwlock_tryrdlock(&ht->user->bucket_locks[h])!= 0){
        send_response(socket, KEY_ERROR, 0, NULL);
        //request->connection_close = 1;
        return -1;
    }

    hash_item_t *item_in_bucket = ht->items[h];

    
    while(item_in_bucket != NULL){
        if(!strcmp(item_in_bucket->key,key)){
            item = item_in_bucket;
            break;
        }
        item_in_bucket = item_in_bucket->next;
    }
    pthread_rwlock_unlock(&ht->user->bucket_locks[h]);

    
    if(item == NULL ){  //key doesn't exist or is locked!!!
        send_response(socket, KEY_ERROR, 0, NULL);
        //request->connection_close = 1;
        return -1;
    }

    
    //fprintf(stderr, "locking %s in GET gave code %d \n",item->key,result_lock);

    if(pthread_rwlock_tryrdlock(&item->user->itemMutex) != 0){
        send_response(socket, KEY_ERROR, 0, NULL);
        //request->connection_close = 1;
        return -1;
    }


    size_t bytes_to_read = item->value_size;   //load value into buffer and send it as response
    char payloadbuf[bytes_to_read];
    memcpy(payloadbuf, item->value,bytes_to_read);

    send_response(socket,OK,bytes_to_read,payloadbuf);  
    //fprintf(stderr, "gonna unlock %s from GET \n",item->key);

    pthread_rwlock_unlock(&(item->user->itemMutex));

    return 0;
}


int del_request(int socket, struct request *request)
{
    //fprintf(stderr,"gonna DEL1 \n");
    hash_item_t *item = NULL;
    char *key = request->key;

    //race conditions need to be inserted so that no concurrent access to bucket, use lock and unlock for item and bucket
    unsigned int h = hash(key) % ht->capacity;   //get bucket

    hash_item_t *item_in_bucket = ht->items[h];

    
    while(item_in_bucket != NULL){
        if(!strcmp(item_in_bucket->key,key)){
            item = item_in_bucket;
            break;
        }
        item_in_bucket = item_in_bucket->next;
    }

    if(item == NULL){  //key doesn't exist or is locked!!!
        send_response(socket, KEY_ERROR, 0, NULL);
        //request->connection_close = 1;
        return -1;
    }

    
    //fprintf(stderr, "locking %s in DEL gave code %d \n",item->key,result_lock);

    if(pthread_rwlock_tryrdlock(&item->user->itemMutex) != 0){
        send_response(socket, KEY_ERROR, 0, NULL);
        //request->connection_close = 1;
        return -1;
    }

    //pthread_rwlock_destroy(&item->user->itemMutex);
    //free(item->value);
    //free(item->key);
    //free(item->user); 

    // update hashtable to remove the item
    if (ht->items[h] == item) {
        ht->items[h] = item->next;
    } 
    if(item->prev == NULL){
        if(item->next == NULL){
            ht->items[h] = NULL;
        }
        else{
            item->next->prev = NULL;
        }
    }
    else if(item->prev != NULL){
        if(item->next != NULL){
            item->next->prev = item->prev;
            item->prev->next = item->next;
        }
    }

    free(item);

    send_response(socket, OK, 0, NULL);

    return 0; 
}

void *main_job(void *arg)
{
    int method;
    struct conn_info *conn_info = arg;
    struct request *request = allocate_request();
    request->connection_close = 0;


    pr_info("Starting new session from %s:%d\n",
        inet_ntoa(conn_info->addr.sin_addr),
        ntohs(conn_info->addr.sin_port));

    do {
        method = recv_request(conn_info->socket_fd, request);
        // Insert your operations here
        switch (method) {
        case SET:
            set_request(conn_info->socket_fd, request);
            break;
        case GET:
            get_request(conn_info->socket_fd, request);
            break;
        case DEL:
            del_request(conn_info->socket_fd, request);
            break;
        case RST:
            // ./check.py issues a reset request after each test
            // to bring back the hashtable to a known state.
            // Implement your reset command here.
            send_response(conn_info->socket_fd, OK, 0, NULL);
            break;
        case STAT:
            break;
        }

        if (request->key) {
            free(request->key);
        }

    } while (!request->connection_close);

    close_connection(conn_info->socket_fd);
    free(request);
    free(conn_info);
    return (void *)NULL;
}

int main(int argc, char *argv[])
{
    int listen_sock;
    pthread_t thread;

    listen_sock = server_init(argc, argv);
    //init hashtable
    ht = malloc(sizeof(hashtable_t));
    ht->items = calloc(HT_CAPACITY,sizeof(hash_item_t*));
    ht->capacity = HT_CAPACITY;
    ht->user = (void *)malloc(sizeof(struct user_ht));

    ht->user->bucket_locks = malloc(ht->capacity * sizeof(pthread_mutex_t));

    // init bucketlocks
    for (unsigned int i = 0; i < ht->capacity; ++i) {
        pthread_rwlock_init(&(ht->user->bucket_locks[i]), NULL);
    }

    for (;;) {
        struct conn_info *conn_info = calloc(1, sizeof(struct conn_info));
        if (accept_new_connection(listen_sock, conn_info) < 0) {
            free(conn_info);
            continue;
        }

        pthread_create(&thread, NULL, main_job, conn_info);
    }

    return 0;
}