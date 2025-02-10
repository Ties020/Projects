#define FUSE_USE_VERSION 26

#include <errno.h>
#include <fuse.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>

#include "sfs.h"
#include "diskio.h"


static const char default_img[] = "test.img";

/* Options passed from commandline arguments */
struct options {
    const char *img;
    int background;
    int verbose;
    int show_help;
    int show_fuse_help;
} options;


#define log(fmt, ...) \
    do { \
        if (options.verbose) \
            printf(" # " fmt, ##__VA_ARGS__); \
    } while (0)


/* libfuse2 leaks, so let's shush LeakSanitizer if we are using Asan. */
const char* __asan_default_options() { return "detect_leaks=0"; }


/*
 * This is a helper function that is optional, but highly recomended you
 * implement and use. Given a path, it looks it up on disk. It will return 0 on
 * success, and a non-zero value on error (e.g., the file did not exist).
 * The resulting directory entry is placed in the memory pointed to by
 * ret_entry. Additionally it can return the offset of that direntry on disk in
 * ret_entry_off, which you can use to update the entry and write it back to
 * disk (e.g., rmdir, unlink, truncate, write).
 *
 * You can start with implementing this function to work just for paths in the
 * root entry, and later modify it to also work for paths with subdirectories.
 * This way, all of your other functions can use this helper and will
 * automatically support subdirectories. To make this function support
 * subdirectories, we recommend you refactor this function to be recursive, and
 * take the current directory as argument as well. For example:
 *
 *  static int get_entry_rec(const char *path, const struct sfs_entry *parent,
 *                           size_t parent_nentries, blockidx_t parent_blockidx,
 *                           struct sfs_entry *ret_entry,
 *                           unsigned *ret_entry_off)
 *
 * Here parent is the directory it is currently searching (at first the rootdir,
 * later the subdir). The parent_nentries tells the function how many entries
 * there are in the directory (SFS_ROOTDIR_NENTRIES or SFS_DIR_NENTRIES).
 * Finally, the parent_blockidx contains the blockidx of the given directory on
 * the disk, which will help in calculating ret_entry_off.
 */

//static int get_entry(const char *path, struct sfs_entry *ret_entry,
  //                   unsigned *ret_entry_off)
//{
    /* Get the next component of the path. Make sure to not modify path if it is
     * the value passed by libfuse (i.e., make a copy). Note that strtok
     * modifies the string you pass it. */

    /* Allocate memory for the rootdir (and later, subdir) and read it from disk */

    /* Loop over all entries in the directory, looking for one with the name
     * equal to the current part of the path. If it is the last part of the
     * path, return it. If there are more parts remaining, recurse to handle
     * that subdirectory. */

    //return -ENOSYS;
//}


/*
 * Retrieve information about a file or directory.
 * You should populate fields of `st` with appropriate information if the
 * file exists and is accessible, or return an error otherwise.
 *
 * For directories, you should at least set st_mode (with S_IFDIR) and st_nlink.
 * For files, you should at least set st_mode (with S_IFREG), st_nlink and
 * st_size.
 *
 * Return 0 on success, < 0 on error.
 */
static int sfs_getattr(const char *path,
                       struct stat *st)
{
    int res = 0;

    log("getattr %s\n", path);

    memset(st, 0, sizeof(struct stat));
    /* Set owner to user/group who mounted the image */
    st->st_uid = getuid();
    st->st_gid = getgid();
    /* Last accessed/modified just now */
    st->st_atime = time(NULL);
    st->st_mtime = time(NULL);
    if (strcmp(path, "/") == 0) {
        st->st_mode = S_IFDIR | 0755;
        st->st_nlink = 2;
        return 0;
    } else {
        struct sfs_entry rootdir[SFS_ROOTDIR_NENTRIES];
        disk_read(rootdir, SFS_ROOTDIR_SIZE, SFS_ROOTDIR_OFF);       

        char *dupe_path = strdup(path);
        const char *top_level_name = strtok(dupe_path,"/");;
        const char *second_level = strtok(NULL, "/");

        if(second_level == NULL){ //just the root level
            for(unsigned int i = 0; i< SFS_ROOTDIR_NENTRIES; i++){
                if (strcmp(rootdir[i].filename, top_level_name) == 0){
                    unsigned int withoutflags = rootdir[i].size & SFS_SIZEMASK; // extract the lower 28 bits (size-mark)
                    unsigned int withflags = rootdir[i].size & SFS_DIRECTORY;  //extract the directory flag

                    if(withoutflags == 0 && withflags != 0 ) {  //it is a directory, since the upper bit is set and the other bits are 0
                        st->st_mode = S_IFDIR | 0755; 
                        st->st_nlink = 2;
                    }
                    else{
                        st->st_mode = S_IFREG | 0755; 
                        st->st_nlink = 2;
                        st->st_size = rootdir[i].size;
                    }
                    
                    return 0;
                }
            }
        
        }
         
        else{
            blockidx_t subdir_offset = 0;

            for(unsigned int i = 0; i< SFS_ROOTDIR_NENTRIES; i++){
                if (strcmp(rootdir[i].filename, top_level_name) == 0){
                    unsigned int withoutflags = rootdir[i].size & SFS_SIZEMASK; // extract the lower 28 bits (size-mark)
                    unsigned int withflags = rootdir[i].size & SFS_DIRECTORY;  //extract the directory flag

                    if(withoutflags == 0 && withflags != 0 ) {  //it is a directory, since the upper bit is set and the other bits are 0
                        st->st_mode = S_IFDIR | 0755; 
                        st->st_nlink = 2;
                        subdir_offset = rootdir[i].first_block;
                    }
                    else{
                        st->st_mode = S_IFREG | 0755; 
                        st->st_nlink = 2;
                        st->st_size = rootdir[i].size;
                    }
                }
            }
            top_level_name = second_level;  
            second_level = strtok(NULL,"/");

            while(top_level_name != NULL){ 
                off_t calc_subdir_off = SFS_DATA_OFF + subdir_offset * SFS_BLOCK_SIZE;
                struct sfs_entry subdir[SFS_DIR_NENTRIES];
                disk_read(subdir, SFS_DIR_SIZE, calc_subdir_off); 
                for(unsigned int i = 0; i < SFS_DIR_NENTRIES; i++){
                    if (strcmp(subdir[i].filename, top_level_name) == 0){
                        if(second_level == NULL){ 
                            unsigned int withoutflags = subdir[i].size & SFS_SIZEMASK; // extract the lower 28 bits (size-mark)
                            unsigned int withflags = subdir[i].size & SFS_DIRECTORY;  //extract the directory flag
    
                            if(withoutflags == 0 && withflags != 0 ) {  //it is a directory, since the upper bit is set and the other bits are 0
                                st->st_mode = S_IFDIR | 0755; 
                                st->st_nlink = 2;
                            }
                            else{
                                st->st_mode = S_IFREG | 0755; 
                                st->st_nlink = 2;
                                st->st_size = subdir[i].size;
                            }
                            return 0;
                        }
                        else{
                            subdir_offset = subdir[i].first_block;
                            break;
                        }
                    }
                }

                top_level_name = second_level;  //go down a level in path
                second_level = strtok(NULL,"/");
            }
            if(res == 0) return 0;
            
        }
        res = -ENOENT;
    }

    return res;
}


/*
 * Return directory contents for `path`. This function should simply fill the
 * filenames - any additional information (e.g., whether something is a file or
 * directory) is later retrieved through getattr calls.
 * Use the function `filler` to add an entry to the directory. Use it like:
 *  filler(buf, <dirname>, NULL, 0);
 * Return 0 on success, < 0 on error.
 */
static int sfs_readdir(const char *path,
                       void *buf,
                       fuse_fill_dir_t filler,
                       off_t offset,
                       struct fuse_file_info *fi)
{
    (void)offset, (void)fi;
    log("readdir %s\n", path);
    if(strcmp(path,"/") == 0) {  //read root directory
        struct sfs_entry rootdir[SFS_ROOTDIR_NENTRIES];
        disk_read(rootdir,SFS_ROOTDIR_SIZE,SFS_ROOTDIR_OFF);
        for(unsigned int i = 0; i < SFS_ROOTDIR_NENTRIES; i++){
            if(strlen(rootdir[i].filename) != 0){  
                filler(buf,rootdir[i].filename,NULL,0);
            }
        }
        filler(buf, ".", NULL, 0);
        filler(buf, "..", NULL, 0);
        return 0;
    }
    else {  //handle subdirectories 
        blockidx_t subdir_offset = 0;
        char *dupe_path = strdup(path);
        
        const char *top_level_name = strtok(dupe_path,"/");;
        const char *second_level = strtok(NULL, "/");

        struct sfs_entry rootdir[SFS_ROOTDIR_NENTRIES];
        disk_read(rootdir,SFS_ROOTDIR_SIZE,SFS_ROOTDIR_OFF);

        for(unsigned int i = 0; i < SFS_ROOTDIR_NENTRIES; i++){
            if(strcmp(rootdir[i].filename,top_level_name) == 0){  
                subdir_offset = rootdir[i].first_block;
                break;
            }
        }

        while(top_level_name != NULL){

            if(second_level == NULL){
                struct sfs_entry subdir[SFS_DIR_NENTRIES];
                off_t calc_subdir_off = SFS_DATA_OFF + subdir_offset * SFS_BLOCK_SIZE;
                disk_read(subdir, SFS_DIR_SIZE, calc_subdir_off);

                for(unsigned int i = 0; i < SFS_DIR_NENTRIES; i++){  //these should all be filled                    
                    if(strlen(subdir[i].filename) != 0){  
                        filler(buf,subdir[i].filename,NULL,0);
                    }
                }
                filler(buf, ".", NULL, 0);
                filler(buf, "..", NULL, 0);

                return 0;
            }
            struct sfs_entry subdir[SFS_DIR_NENTRIES];
            off_t calc_subdir_off = SFS_DATA_OFF + subdir_offset * SFS_BLOCK_SIZE;
            disk_read(subdir, SFS_DIR_SIZE, calc_subdir_off); 

            for(unsigned int i = 0; i < SFS_DIR_NENTRIES; i++){  //these should all be filled
                if(strcmp(subdir[i].filename, second_level) == 0){
                    subdir_offset = subdir[i].first_block;
                    break;
                }
            }
            
            top_level_name = second_level;
            second_level = strtok(NULL,"/");
        }
        filler(buf, ".", NULL, 0);
        filler(buf, "..", NULL, 0);

        return 0;
    }
    return -ENOENT;

}


/*
 * Read contents of `path` into `buf` for  up to `size` bytes.
 * Note that `size` may be bigger than the file actually is.
 * Reading should start at offset `offset`; the OS will generally read your file
 * in chunks of 4K byte.
 * Returns the number of bytes read (writtin into `buf`), or < 0 on error.
 */
static int sfs_read(const char *path,
                    char *buf,
                    size_t size,
                    off_t offset,
                    struct fuse_file_info *fi)
{
    (void)fi;
    log("read %s size=%zu offset=%ld\n", path, size, offset);

    struct sfs_entry rootdir[SFS_ROOTDIR_NENTRIES];
    disk_read(rootdir, SFS_ROOTDIR_SIZE, SFS_ROOTDIR_OFF);       
    char *dupe_path = strdup(path);
    const char *top_name = strtok(dupe_path,"/");
    const char *second_name = strtok(NULL,"/");
    blockidx_t init_block = 0;

    for(unsigned int i = 0; i < SFS_ROOTDIR_NENTRIES; i++){
        if(strcmp(rootdir[i].filename,top_name) == 0){  
            init_block = rootdir[i].first_block;
            break;
        }
    }

    while(top_name != NULL){
        if(second_name == NULL){
            size_t bytes_read = 0;
            off_t current_offset = 0;
            blockidx_t blockindex = init_block;
            size_t remaining_size = size;

            blockidx_t blocktable[SFS_BLOCKTBL_NENTRIES]; 
            disk_read(blocktable, SFS_BLOCKTBL_SIZE, SFS_BLOCKTBL_OFF); 


           if(offset == 0){
                bytes_read = 0;
                current_offset = SFS_DATA_OFF + init_block * SFS_BLOCK_SIZE;
                blockindex = init_block;
                remaining_size = size;
            }

            if (blocktable[blockindex] == SFS_BLOCKIDX_END){  //if it's a small file
                size_t bytes_to_copy = size;
                disk_read(buf, bytes_to_copy, current_offset);
                return bytes_to_copy;
            }

            while (blockindex != SFS_BLOCKIDX_END) {
                size_t bytes_to_copy = SFS_BLOCK_SIZE;
                if(remaining_size < bytes_to_copy){
                    bytes_to_copy = remaining_size;
                }
                char read_buffer[bytes_to_copy];
                disk_read(read_buffer, bytes_to_copy, current_offset);
                memcpy(buf + bytes_read, read_buffer, bytes_to_copy);
                
                blockindex = blocktable[blockindex];
                remaining_size -= bytes_to_copy;
                bytes_read += bytes_to_copy;
                current_offset = SFS_DATA_OFF + blockindex * SFS_BLOCK_SIZE;
                log("after offset %u\n", current_offset);

            }
            return bytes_read; // Return the number of bytes read
        }

        struct sfs_entry subdir[SFS_DIR_NENTRIES];
        off_t calc_subdir_off = SFS_DATA_OFF + init_block * SFS_BLOCK_SIZE;
        disk_read(subdir, SFS_DIR_SIZE, calc_subdir_off); 

        for(unsigned int i = 0; i < SFS_DIR_NENTRIES; i++){ 
            if(strcmp(subdir[i].filename, second_name) == 0){
                init_block = subdir[i].first_block;
                break;
            }
        }
        
        top_name = second_name;
        second_name = strtok(NULL,"/");
    }
    
    

    (void)buf; /* Placeholder - use me */

    return -ENOSYS;
}


/*
 * Create directory at `path`.
 * The `mode` argument describes the permissions, which you may ignore for this
 * assignment.
 * Returns 0 on success, < 0 on error.
 */
static int sfs_mkdir(const char *path,
                     mode_t mode)
{
    log("mkdir %s mode=%o\n", path, mode);

    return -ENOSYS;
}


/*
 * Remove directory at `path`.
 * Directories may only be removed if they are empty, otherwise this function
 * should return -ENOTEMPTY.
 * Returns 0 on success, < 0 on error.
 */
static int sfs_rmdir(const char *path)
{
    log("rmdir %s\n", path);

    struct sfs_entry rootdir[SFS_ROOTDIR_NENTRIES];
    disk_read(rootdir,SFS_ROOTDIR_SIZE,SFS_ROOTDIR_OFF);
    char *dupe_path = strdup(path);
    const char *top_name = strtok(dupe_path,"/");
    for(unsigned int i = 0; i < SFS_ROOTDIR_NENTRIES; i++){
        if(strcmp(rootdir[i].filename,top_name) == 0){
            blockidx_t init_block = rootdir[i].first_block;
            bool isempty = true;
            struct sfs_entry subdir[SFS_DIR_NENTRIES];
            off_t calc_subdir_off = SFS_DATA_OFF + init_block * SFS_BLOCK_SIZE;
            disk_read(subdir, SFS_DIR_SIZE, calc_subdir_off); 
            for(unsigned int j = 0; j < SFS_DIR_NENTRIES; j++){ 
                if(strlen(subdir[j].filename) != 0){
                    log("isnt empty");
                    isempty = false;
                }
            }

            if(isempty){
                blockidx_t blocktable[SFS_BLOCKTBL_NENTRIES]; 
                disk_read(blocktable, SFS_BLOCKTBL_SIZE, SFS_BLOCKTBL_OFF); 

                off_t offset_blk = SFS_BLOCKTBL_OFF + (rootdir[i].first_block * sizeof(blockidx_t));
                blockidx_t secblock = blocktable[rootdir[i].first_block];
                blocktable[rootdir[i].first_block] = SFS_BLOCKIDX_EMPTY;
                disk_write(&blocktable[rootdir[i].first_block], sizeof(blockidx_t), offset_blk); 

                off_t newoff = SFS_BLOCKTBL_OFF + (secblock * sizeof(blockidx_t));
                blocktable[secblock] = SFS_BLOCKIDX_EMPTY;
                disk_write(&blocktable[secblock], sizeof(blockidx_t), newoff); 

                memset(rootdir[i].filename, '\0', sizeof(rootdir[i].filename));
                rootdir[i].size = 0;
                rootdir[i].first_block = SFS_BLOCKIDX_EMPTY;

                off_t off = SFS_ROOTDIR_OFF + (sizeof(struct sfs_entry) * i);
                disk_write(&rootdir[i], sizeof(struct sfs_entry), off); //write modified entry back to disk

                return 0;
            }
            else {
                return -ENOTEMPTY;
            }
        }

    }

    return -ENOSYS;
}
/*
 * Remove file at `path`.
 * Can not be used to remove directories.
 * Returns 0 on success, < 0 on error.
 */
static int sfs_unlink(const char *path)
{
    log("unlink %s\n", path);
    struct sfs_entry rootdir[SFS_ROOTDIR_NENTRIES];
    disk_read(rootdir,SFS_ROOTDIR_SIZE,SFS_ROOTDIR_OFF);
    char *dupe_path = strdup(path);
    const char *top_name = strtok(dupe_path,"/");
    log("NAME SI %s\n",top_name);

    for(unsigned int i = 0; i < SFS_ROOTDIR_NENTRIES; i++){
        if(strcmp(rootdir[i].filename,top_name) == 0){
            memset(rootdir[i].filename, '\0', sizeof(rootdir[i].filename));
            rootdir[i].size = 0;
            rootdir[i].first_block = SFS_BLOCKIDX_EMPTY;

            off_t off = SFS_ROOTDIR_OFF + (sizeof(struct sfs_entry) * i);
            disk_write(&rootdir[i], sizeof(struct sfs_entry), off); //write modified entry back to disk

            return 0;

        }
    }

    return -ENOSYS;
}


/*
 * Create an empty file at `path`.
 * The `mode` argument describes the permissions, which you may ignore for this
 * assignment.
 * Returns 0 on success, < 0 on error.
 */
static int sfs_create(const char *path,
                      mode_t mode,
                      struct fuse_file_info *fi)
{
    (void)fi;
    log("create %s mode=%o\n", path, mode);

    log("path is  %s\n", path);

    

    //use disk_write, fill buff with 0s, then write it to offset calculated by i 
    //or just set size to 0

    return -ENOSYS;
}


/*
 * Shrink or grow the file at `path` to `size` bytes.
 * Excess bytes are thrown away, whereas any bytes added in the process should
 * be nil (\0).
 * Returns 0 on success, < 0 on error.
 */
static int sfs_truncate(const char *path, off_t size)
{
    log("truncate %s size=%ld\n", path, size);


    return -ENOSYS;
}


/*
 * Write contents of `buf` (of `size` bytes) to the file at `path`.
 * The file is grown if nessecary, and any bytes already present are overwritten
 * (whereas any other data is left intact). The `offset` argument specifies how
 * many bytes should be skipped in the file, after which `size` bytes from
 * buffer are written.
 * This means that the new file size will be max(old_size, offset + size).
 * Returns the number of bytes written, or < 0 on error.
 */
static int sfs_write(const char *path,
                     const char *buf,
                     size_t size,
                     off_t offset,
                     struct fuse_file_info *fi)
{
    (void)fi;
    log("write %s data='%.*s' size=%zu offset=%ld\n", path, (int)size, buf,
        size, offset);

    return -ENOSYS;
}


/*
 * Move/rename the file at `path` to `newpath`.
 * Returns 0 on succes, < 0 on error.
 */
static int sfs_rename(const char *path,
                      const char *newpath)
{
    /* Implementing this function is optional, and not worth any points. */
    log("rename %s %s\n", path, newpath);

    return -ENOSYS;
}


static const struct fuse_operations sfs_oper = {
    .getattr    = sfs_getattr,
    .readdir    = sfs_readdir,
    .read       = sfs_read,
    .mkdir      = sfs_mkdir,
    .rmdir      = sfs_rmdir,
    .unlink     = sfs_unlink,
    .create     = sfs_create,
    .truncate   = sfs_truncate,
    .write      = sfs_write,
    .rename     = sfs_rename,
};


#define OPTION(t, p)                            \
    { t, offsetof(struct options, p), 1 }
#define LOPTION(s, l, p)                        \
    OPTION(s, p),                               \
    OPTION(l, p)
static const struct fuse_opt option_spec[] = {
    LOPTION("-i %s",    "--img=%s",     img),
    LOPTION("-b",       "--background", background),
    LOPTION("-v",       "--verbose",    verbose),
    LOPTION("-h",       "--help",       show_help),
    OPTION(             "--fuse-help",  show_fuse_help),
    FUSE_OPT_END
};

static void show_help(const char *progname)
{
    printf("usage: %s mountpoint [options]\n\n", progname);
    printf("By default this FUSE runs in the foreground, and will unmount on\n"
           "exit. If something goes wrong and FUSE does not exit cleanly, use\n"
           "the following command to unmount your mountpoint:\n"
           "  $ fusermount -u <mountpoint>\n\n");
    printf("common options (use --fuse-help for all options):\n"
           "    -i, --img=FILE      filename of SFS image to mount\n"
           "                        (default: \"%s\")\n"
           "    -b, --background    run fuse in background\n"
           "    -v, --verbose       print debug information\n"
           "    -h, --help          show this summarized help\n"
           "        --fuse-help     show full FUSE help\n"
           "\n", default_img);
}

int main(int argc, char **argv)
{
    struct fuse_args args = FUSE_ARGS_INIT(argc, argv);

    options.img = strdup(default_img);

    fuse_opt_parse(&args, &options, option_spec, NULL);

    if (options.show_help) {
        show_help(argv[0]);
        return 0;
    }

    if (options.show_fuse_help) {
        assert(fuse_opt_add_arg(&args, "--help") == 0);
        args.argv[0][0] = '\0';
    }

    if (!options.background)
        assert(fuse_opt_add_arg(&args, "-f") == 0);

    disk_open_image(options.img);

    return fuse_main(args.argc, args.argv, &sfs_oper, NULL);
}