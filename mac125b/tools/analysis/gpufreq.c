// GPU performance-state residency over an interval (IOReport, no sudo).
// build: clang -O2 gpufreq.c -o gpufreq -framework CoreFoundation -lIOReport
// usage: gpufreq <seconds> [interval_ms]   -> prints residency % per GPU P-state for each interval
#include <CoreFoundation/CoreFoundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct IOReportSubscriptionRef * IOReportSubscriptionRef;
extern CFMutableDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
extern IOReportSubscriptionRef IOReportCreateSubscription(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern CFStringRef IOReportChannelGetGroup(CFDictionaryRef);
extern CFStringRef IOReportChannelGetSubGroup(CFDictionaryRef);
extern CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
extern int IOReportStateGetCount(CFDictionaryRef);
extern CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef, int);
extern int64_t IOReportStateGetResidency(CFDictionaryRef, int);

static void cs(CFStringRef s, char * buf, int n) {
    buf[0] = 0;
    if (s) CFStringGetCString(s, buf, n, kCFStringEncodingUTF8);
}

int main(int argc, char ** argv) {
    const double secs = argc > 1 ? atof(argv[1]) : 10.0;
    const int iv = argc > 2 ? atoi(argv[2]) : 1000;
    CFMutableDictionaryRef ch = IOReportCopyChannelsInGroup(CFSTR("GPU Stats"), NULL, 0, 0, 0);
    if (!ch) { fprintf(stderr, "no GPU Stats channels\n"); return 1; }
    CFMutableDictionaryRef subbed = NULL;
    IOReportSubscriptionRef sub = IOReportCreateSubscription(NULL, ch, &subbed, 0, NULL);
    if (!sub) { fprintf(stderr, "subscription failed\n"); return 1; }
    CFDictionaryRef prev = IOReportCreateSamples(sub, subbed, NULL);
    const int n = (int) (secs*1000/iv);
    for (int it = 0; it < n; ++it) {
        usleep(iv*1000);
        CFDictionaryRef cur = IOReportCreateSamples(sub, subbed, NULL);
        CFDictionaryRef d = IOReportCreateSamplesDelta(prev, cur, NULL);
        CFRelease(prev); prev = cur;
        CFArrayRef arr = CFDictionaryGetValue(d, CFSTR("IOReportChannels"));
        for (CFIndex i = 0; arr && i < CFArrayGetCount(arr); ++i) {
            CFDictionaryRef c = CFArrayGetValueAtIndex(arr, i);
            char sg[128], nm[128];
            cs(IOReportChannelGetSubGroup(c), sg, sizeof sg);
            cs(IOReportChannelGetChannelName(c), nm, sizeof nm);
            if (strcmp(sg, "GPU Performance States") != 0) continue;
            const int ns = IOReportStateGetCount(c);
            int64_t tot = 0;
            for (int k = 0; k < ns; ++k) tot += IOReportStateGetResidency(c, k);
            if (tot <= 0) continue;
            printf("t=%d %s:", it, nm);
            for (int k = 0; k < ns; ++k) {
                const int64_t r = IOReportStateGetResidency(c, k);
                if (r*1000 < tot) continue;
                char sn[64]; cs(IOReportStateGetNameForIndex(c, k), sn, sizeof sn);
                printf(" %s=%.1f%%", sn, 100.0*r/tot);
            }
            printf("\n");
        }
        CFRelease(d);
        fflush(stdout);
    }
    return 0;
}
