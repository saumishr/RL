// testserver is a local HTTP/2 origin that withholds its response for a fixed
// delay, standing in for a slow NVCF inference call. It exists to prove the
// sidecar keeps a read-idle connection alive with PING frames, which cannot be
// verified against NVCF without a live API key.
package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"flag"
	"fmt"
	"log"
	"math/big"
	"net"
	"net/http"
	"time"
)

func selfSigned() tls.Certificate {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Fatalf("key: %v", err)
	}
	tmpl := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "localhost"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IPAddresses:  []net.IP{net.ParseIP("127.0.0.1")},
		DNSNames:     []string{"localhost"},
		IsCA:         true,
	}
	der, err := x509.CreateCertificate(rand.Reader, &tmpl, &tmpl, &key.PublicKey, key)
	if err != nil {
		log.Fatalf("cert: %v", err)
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}
}

func main() {
	addr := flag.String("listen", "127.0.0.1:8443", "HTTPS listen address")
	delay := flag.Duration("delay", 20*time.Second, "withhold the response for this long")
	flag.Parse()

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		log.Printf("request %s %s proto=%s; withholding response for %v", r.Method, r.URL.Path, r.Proto, *delay)
		select {
		case <-time.After(*delay):
		case <-r.Context().Done():
			log.Printf("client vanished after %v: %v", time.Since(start).Truncate(time.Millisecond), r.Context().Err())
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"ok":true,"proto":%q,"delayed_s":%.1f}`+"\n", r.Proto, time.Since(start).Seconds())
		log.Printf("responded after %v", time.Since(start).Truncate(time.Millisecond))
	})

	server := &http.Server{
		Addr:    *addr,
		Handler: mux,
		TLSConfig: &tls.Config{
			Certificates: []tls.Certificate{selfSigned()},
			NextProtos:   []string{"h2"},
		},
	}
	log.Printf("h2 test origin on %s, delay=%v", *addr, *delay)
	log.Fatal(server.ListenAndServeTLS("", ""))
}
