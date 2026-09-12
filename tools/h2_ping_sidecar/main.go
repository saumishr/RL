// h2-ping-sidecar proxies HTTP/1.1 requests onto an HTTP/2 connection that
// emits PING frames while waiting for a response.
//
// NVCF's endpoints sit behind AWS Global Accelerator, which drops any TCP
// connection carrying no application-layer data for 340s. That limit is fixed
// by AWS. TCP keepalive does not reset it because only bytes above the TCP
// layer count, so a non-streaming inference request that needs more than 340s
// to produce its first response byte can never succeed. HTTP/2 PING frames are
// application data and do reset the timer, but aiohttp has no HTTP/2 support at
// all, so the client cannot send them itself. This sidecar owns the upstream
// connection instead and pings it on the client's behalf.
package main

import (
	"context"
	"crypto/tls"
	"errors"
	"flag"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
)

func main() {
	listen := flag.String("listen", "127.0.0.1:8080", "local HTTP/1.1 listen address")
	upstream := flag.String("upstream", "", "HTTPS base URL to forward to (required)")
	pingInterval := flag.Duration("ping-interval", 60*time.Second, "send a PING after this much read-idle time; must stay well below 340s")
	pingTimeout := flag.Duration("ping-timeout", 15*time.Second, "close the upstream connection if a PING is not acknowledged within this time")
	shutdownGrace := flag.Duration("shutdown-grace", 15*time.Minute, "on SIGTERM, wait up to this long for in-flight requests")
	insecure := flag.Bool("insecure-skip-verify", false, "skip upstream TLS verification (testing only)")
	readyFile := flag.String("ready-file", "", "write this file once the listener is bound, so a supervisor can distinguish our listener from a port already in use by something else")
	flag.Parse()

	if *upstream == "" {
		log.Fatal("-upstream is required")
	}
	target, err := url.Parse(*upstream)
	if err != nil {
		log.Fatalf("-upstream is not a valid URL: %v", err)
	}
	if target.Scheme != "https" {
		log.Fatalf("-upstream must be an https:// URL (got %q); HTTP/2 is negotiated via ALPN", target.Scheme)
	}
	if *pingInterval >= 340*time.Second {
		log.Fatalf("-ping-interval %v is at or above the 340s Global Accelerator limit it exists to defeat", *pingInterval)
	}

	// HTTP/2 only. Falling back to HTTP/1.1 would silently reintroduce the
	// 340s failure, since PING frames do not exist in HTTP/1.1.
	protocols := new(http.Protocols)
	protocols.SetHTTP2(true)

	transport := &http.Transport{
		Protocols: protocols,
		HTTP2: &http.HTTP2Config{
			SendPingTimeout: *pingInterval,
			PingTimeout:     *pingTimeout,
		},
		TLSClientConfig:     &tls.Config{InsecureSkipVerify: *insecure},
		TLSHandshakeTimeout: 30 * time.Second,
		DialContext:         (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		// ResponseHeaderTimeout is deliberately unset. Waiting a long time for
		// the first response byte is the entire purpose of this proxy; the
		// client decides its own deadline.
	}

	proxy := &httputil.ReverseProxy{
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(target)
			// Empty Host makes net/http derive it from the outbound URL, so
			// upstream sees its own hostname rather than the loopback address.
			r.Out.Host = ""
		},
		Transport: transport,
		// Flush immediately so streaming (SSE) responses are not buffered.
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if errors.Is(err, context.Canceled) {
				return
			}
			log.Printf("upstream error: %s %s: %v", r.Method, r.URL.Path, err)
			w.WriteHeader(http.StatusBadGateway)
			// Body carries the Go error; headers are never logged or echoed.
			_, _ = w.Write([]byte("h2-ping-sidecar: upstream error: " + err.Error() + "\n"))
		},
	}

	server := &http.Server{
		Addr:    *listen,
		Handler: proxy,
		// No read or write timeouts on the client-facing side: long requests
		// are the point.
	}

	idle := make(chan struct{})
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
		<-sig
		log.Printf("shutdown requested; draining in-flight requests for up to %v", *shutdownGrace)
		ctx, cancel := context.WithTimeout(context.Background(), *shutdownGrace)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			log.Printf("graceful shutdown incomplete: %v", err)
		}
		close(idle)
	}()

	// Bind before signalling readiness. A supervisor that probes the port
	// instead cannot tell our listener from an unrelated process that already
	// holds it, and would let clients send traffic to the wrong server.
	listener, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Fatalf("cannot bind %s: %v", *listen, err)
	}
	if *readyFile != "" {
		if err := os.WriteFile(*readyFile, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644); err != nil {
			log.Fatalf("cannot write -ready-file %s: %v", *readyFile, err)
		}
		defer os.Remove(*readyFile)
	}

	log.Printf("listening on %s -> %s (HTTP/2, PING every %v idle, ack deadline %v)",
		listener.Addr(), target.String(), *pingInterval, *pingTimeout)
	if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("serve failed: %v", err)
	}
	<-idle
	log.Print("stopped")
}
