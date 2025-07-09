#ifndef HELPER_MULTIPROCESS_H
#define HELPER_MULTIPROCESS_H

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

// Shared memory structure
typedef struct {
    HANDLE hMapFile;
    void* addr;
} sharedMemoryInfo;

// Process structure
typedef struct {
    PROCESS_INFORMATION pi;
    HANDLE hProcess;
} Process;

// Create shared memory
static int sharedMemoryCreate(const char* name, size_t size, sharedMemoryInfo* info) {
    info->hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        (DWORD)size,
        name
    );
    
    if (info->hMapFile == NULL) {
        return -1;
    }
    
    info->addr = MapViewOfFile(
        info->hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        size
    );
    
    if (info->addr == NULL) {
        CloseHandle(info->hMapFile);
        return -1;
    }
    
    return 0;
}

// Open shared memory
static int sharedMemoryOpen(const char* name, size_t size, sharedMemoryInfo* info) {
    info->hMapFile = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        name
    );
    
    if (info->hMapFile == NULL) {
        return -1;
    }
    
    info->addr = MapViewOfFile(
        info->hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        size
    );
    
    if (info->addr == NULL) {
        CloseHandle(info->hMapFile);
        return -1;
    }
    
    return 0;
}

// Close shared memory
static void sharedMemoryClose(sharedMemoryInfo* info) {
    if (info->addr) {
        UnmapViewOfFile(info->addr);
        info->addr = NULL;
    }
    if (info->hMapFile) {
        CloseHandle(info->hMapFile);
        info->hMapFile = NULL;
    }
}

// Spawn process
static int spawnProcess(Process* process, const char* app, char* const args[]) {
    STARTUPINFOA si;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    memset(&process->pi, 0, sizeof(process->pi));
    
    // Build command line
    char cmdLine[1024] = {0};
    strcat(cmdLine, app);
    for (int i = 1; args[i] != NULL; i++) {
        strcat(cmdLine, " ");
        strcat(cmdLine, args[i]);
    }
    
    BOOL success = CreateProcessA(
        NULL,           // No module name (use command line)
        cmdLine,        // Command line
        NULL,           // Process handle not inheritable
        NULL,           // Thread handle not inheritable
        FALSE,          // Set handle inheritance to FALSE
        0,              // No creation flags
        NULL,           // Use parent's environment block
        NULL,           // Use parent's starting directory
        &si,            // Pointer to STARTUPINFO structure
        &process->pi    // Pointer to PROCESS_INFORMATION structure
    );
    
    if (!success) {
        return -1;
    }
    
    process->hProcess = process->pi.hProcess;
    return 0;
}

// Wait for process
static int waitProcess(Process* process) {
    DWORD exitCode = 0;
    
    WaitForSingleObject(process->hProcess, INFINITE);
    GetExitCodeProcess(process->hProcess, &exitCode);
    
    CloseHandle(process->pi.hProcess);
    CloseHandle(process->pi.hThread);
    
    return (int)exitCode;
}

#endif // HELPER_MULTIPROCESS_H