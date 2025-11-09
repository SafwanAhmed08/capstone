/* iiot_mitm_simple.cc
   ns-3 v3.35
   Simple IIoT topology + minimal DNS MITM (spoofing) demo.

   - IoT CSMA LAN with nIoT nodes (clients)
   - Inline node (acts as local DNS resolver + MITM)
   - Gateway node
   - Server node (real DNS backend)
   - IoT nodes send TCP telemetry (BulkSend) and DNS queries to inline resolver
   - Inline resolver normally forwards queries to upstream server and relays responses
   - During attack window (attackStart..attackStop), inline sends spoofed DNS responses (attacker IP)
   - Writes meta_mitm.json with attack metadata
*/

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/csma-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("IIoT_MITM_Simple");

static const uint16_t DNS_PORT = 53;

// Forward declarations
void SendPacketTo(Ptr<Socket> sock, std::vector<uint8_t> q);
void ServerDnsRecv(Ptr<Socket> sock);
void InlineDnsRecv(Ptr<Socket> sock);
void SendDnsQuery(Ptr<Socket> sock, uint16_t txid, uint16_t qtype);

// Global variables for callback functions
std::map<uint16_t, Address> g_pending;
double g_attackStart;
double g_attackStop;
std::string g_spoofIp;
Ipv4Address g_serverAddr;

// Build a simple DNS query for "example.com" with specified QTYPE
std::vector<uint8_t> 
BuildDnsQuery(uint16_t txid = 0x1a2b, uint16_t qtype = 1)
{
  std::vector<uint8_t> q;
  // DNS header (12 bytes)
  q.push_back(static_cast<uint8_t>((txid >> 8) & 0xFF));
  q.push_back(static_cast<uint8_t>(txid & 0xFF));
  q.push_back(0x01); q.push_back(0x00); // flags: standard query (0x0100)
  q.push_back(0x00); q.push_back(0x01); // QDCOUNT = 1
  q.push_back(0x00); q.push_back(0x00); // ANCOUNT = 0
  q.push_back(0x00); q.push_back(0x00); // NSCOUNT = 0
  q.push_back(0x00); q.push_back(0x00); // ARCOUNT = 0

  // Question: "example.com" in label format
  const std::string name = "example.com";
  size_t pos = 0;
  while (pos < name.size())
  {
    size_t dot = name.find('.', pos);
    std::string label = (dot == std::string::npos) ? name.substr(pos) : name.substr(pos, dot - pos);
    q.push_back(static_cast<uint8_t>(label.size()));
    for (char c : label) q.push_back(static_cast<uint8_t>(c));
    if (dot == std::string::npos) break;
    pos = dot + 1;
  }
  q.push_back(0x00); // end of qname

  // QTYPE (variable), QCLASS = IN (1)
  q.push_back(static_cast<uint8_t>((qtype >> 8) & 0xFF));
  q.push_back(static_cast<uint8_t>(qtype & 0xFF));
  q.push_back(0x00); q.push_back(0x01);

  return q;
}

// Get a random DNS query type (A, AAAA, MX, TXT)
uint16_t
GetRandomQueryType()
{
  static const uint16_t qtypes[] = {
    1,   // A record
    28,  // AAAA record (IPv6)
    15,  // MX record
    16   // TXT record
  };
  int index = std::rand() % 4;
  return qtypes[index];
}

// Send helper used by Schedule
void
SendDnsQuery(Ptr<Socket> sock, uint16_t txid, uint16_t qtype)
{
  if (sock == nullptr) return;
  std::vector<uint8_t> query = BuildDnsQuery(txid, qtype);
  Ptr<Packet> p = Create<Packet>(query.data(), query.size());
  sock->Send(p);
}

std::vector<uint8_t> BuildDnsResponseWithAnswer(uint16_t txid, const std::string &qname, uint16_t qtype, const std::string &answerIp)
{
  // Build a minimal DNS response with one A record if qtype == 1 (A), otherwise leave ANCOUNT=0.
  std::vector<uint8_t> r;
  // Header
  r.push_back(static_cast<uint8_t>((txid >> 8) & 0xFF));
  r.push_back(static_cast<uint8_t>(txid & 0xFF));
  r.push_back(0x81); r.push_back(0x80); // flags: response, no error
  r.push_back(0x00); r.push_back(0x01); // QDCOUNT = 1
  if (qtype == 1) { r.push_back(0x00); r.push_back(0x01); } else { r.push_back(0x00); r.push_back(0x00); } // ANCOUNT
  r.push_back(0x00); r.push_back(0x00); // NSCOUNT
  r.push_back(0x00); r.push_back(0x00); // ARCOUNT

  // Question: encode qname labels
  size_t pos = 0;
  while (pos < qname.size())
  {
    size_t dot = qname.find('.', pos);
    std::string label = (dot == std::string::npos) ? qname.substr(pos) : qname.substr(pos, dot - pos);
    r.push_back(static_cast<uint8_t>(label.size()));
    for (char c : label) r.push_back(static_cast<uint8_t>(c));
    if (dot == std::string::npos) break;
    pos = dot + 1;
  }
  r.push_back(0x00); // end qname
  // QTYPE, QCLASS
  r.push_back(static_cast<uint8_t>((qtype >> 8) & 0xFF)); r.push_back(static_cast<uint8_t>(qtype & 0xFF));
  r.push_back(0x00); r.push_back(0x01);

  if (qtype == 1)
  {
    // Answer: NAME pointer to question (0xC00C), TYPE A (1), CLASS IN (1), TTL, RDLENGTH, RDATA
    r.push_back(0xC0); r.push_back(0x0C);
    r.push_back(0x00); r.push_back(0x01); // TYPE A
    r.push_back(0x00); r.push_back(0x01); // CLASS IN
    r.push_back(0x00); r.push_back(0x00); r.push_back(0x00); r.push_back(0x3C); // TTL 60
    r.push_back(0x00); r.push_back(0x04); // RDLENGTH 4
    // parse answerIp as dotted quad
    std::stringstream ss(answerIp);
    uint32_t b[4]; char dot;
    ss >> b[0] >> dot >> b[1] >> dot >> b[2] >> dot >> b[3];
    r.push_back(static_cast<uint8_t>(b[0]));
    r.push_back(static_cast<uint8_t>(b[1]));
    r.push_back(static_cast<uint8_t>(b[2]));
    r.push_back(static_cast<uint8_t>(b[3]));
  }

  return r;
}

// Helper: parse txid (first two bytes of DNS message) and whether response bit is set
void ParseDnsHeader(const std::vector<uint8_t> &buf, uint16_t &txid, bool &isResponse, uint16_t &qdcount, uint16_t &ancount)
{
  txid = 0;
  isResponse = false;
  qdcount = 0; ancount = 0;
  if (buf.size() < 12) return;
  txid = (static_cast<uint16_t>(buf[0]) << 8) | static_cast<uint16_t>(buf[1]);
  uint16_t flags = (static_cast<uint16_t>(buf[2]) << 8) | static_cast<uint16_t>(buf[3]);
  isResponse = (flags & 0x8000) != 0;
  qdcount = (static_cast<uint16_t>(buf[4]) << 8) | static_cast<uint16_t>(buf[5]);
  ancount = (static_cast<uint16_t>(buf[6]) << 8) | static_cast<uint16_t>(buf[7]);
}

// Helper: extract qname and qtype from query buffer. Works for basic single-question queries.
bool ExtractQueryNameAndType(const std::vector<uint8_t> &buf, std::string &qname, uint16_t &qtype)
{
  if (buf.size() < 12) return false;
  size_t i = 12;
  std::string name;
  while (i < buf.size())
  {
    uint8_t len = buf[i];
    if (len == 0) { ++i; break; }
    if (i + 1 + len > buf.size()) return false;
    if (!name.empty()) name.push_back('.');
    for (uint8_t k = 0; k < len; ++k) name.push_back(static_cast<char>(buf[i+1+k]));
    i += 1 + len;
  }
  if (i + 3 >= buf.size()) return false;
  qtype = (static_cast<uint16_t>(buf[i]) << 8) | static_cast<uint16_t>(buf[i+1]);
  qname = name;
  return true;
}

// Server DNS receive callback function
void ServerDnsRecv(Ptr<Socket> sock)
{
  Address from;
  Ptr<Packet> p = sock->RecvFrom(from);
  if (!p) return;
  uint32_t sz = p->GetSize();
  std::vector<uint8_t> buf(sz);
  p->CopyData(buf.data(), sz);

  uint16_t txid; bool isResp; uint16_t qdcount, ancount;
  ParseDnsHeader(buf, txid, isResp, qdcount, ancount);
  if (isResp) return; // ignore responses
  std::string qname; uint16_t qtype;
  bool ok = ExtractQueryNameAndType(buf, qname, qtype);
  if (!ok) return;
  
  // Build response based on query type (only A records get answers, others get empty response)
  std::string a_ip = "192.0.2.1";
  std::vector<uint8_t> resp = BuildDnsResponseWithAnswer(txid, qname, qtype, a_ip);
  Ptr<Packet> rp = Create<Packet>(resp.data(), resp.size());
  sock->SendTo(rp, 0, from);
  // log with query type info
  NS_LOG_INFO("Server: responded to query " << qname << " id=" << txid << " qtype=" << qtype);
}

// Inline DNS receive callback function
void InlineDnsRecv(Ptr<Socket> sock)
{
  Address from;
  Ptr<Packet> p = sock->RecvFrom(from);
  if (!p) return;
  uint32_t sz = p->GetSize();
  std::vector<uint8_t> buf(sz);
  p->CopyData(buf.data(), sz);

  uint16_t txid; bool isResp; uint16_t qdcount, ancount;
  ParseDnsHeader(buf, txid, isResp, qdcount, ancount);

  double now = Simulator::Now().GetSeconds();
  if (!isResp)
  {
    // It's a query from a client.
    std::string qname; uint16_t qtype;
    bool ok = ExtractQueryNameAndType(buf, qname, qtype);
    if (!ok) return;

    // If inside attack window, craft a spoofed response and send it directly to client
    if (now >= g_attackStart && now <= g_attackStop)
    {
      NS_LOG_INFO("INLINE MITM: Spoofing query '" << qname << "' (txid=" << txid << ") for client " << InetSocketAddress::ConvertFrom(from).GetIpv4());
      std::vector<uint8_t> resp = BuildDnsResponseWithAnswer(txid, qname, qtype, g_spoofIp);
      Ptr<Packet> rp = Create<Packet>(resp.data(), resp.size());
      sock->SendTo(rp, 0, from); // reply to client
      // Do NOT forward this query upstream (we simulate on-path spoofing)
      return;
    }

    // Otherwise (normal), forward query to upstream server and remember client
    g_pending[txid] = from;
    InetSocketAddress upstream = InetSocketAddress(g_serverAddr, DNS_PORT);
    Ptr<Packet> fp = Create<Packet>(buf.data(), buf.size());
    // send to upstream server
    sock->SendTo(fp, 0, upstream);
    NS_LOG_INFO("INLINE: forwarded query '" << qname << "' id=" << txid << " to upstream");
  }
  else
  {
    // It's a response from upstream server targeted at inline (we forwarded earlier)
    // Look up which client asked for this txid
    auto it = g_pending.find(txid);
    if (it != g_pending.end())
    {
      Address clientAddr = it->second;
      Ptr<Packet> rp = Create<Packet>(buf.data(), buf.size());
      sock->SendTo(rp, 0, clientAddr); // send upstream response back to original client
      g_pending.erase(it);
      NS_LOG_INFO("INLINE: relayed response id=" << txid << " back to client " << InetSocketAddress::ConvertFrom(clientAddr).GetIpv4());
    }
  }
}

int main (int argc, char *argv[])
{
  // Parameters
  uint32_t nIoT = 4;
  double simTime = 30.0;
  uint16_t tcpPort = 5001;
  uint32_t maxBytes = 10240;
  bool enablePcap = true;
  bool enableFlowMonitor = true;
  double dnsInterval = 5.0;
  double attackStart = 10.0;
  double attackStop = 15.0;
  std::string spoofIp = "10.1.2.200"; // attacker IP to place in spoofed A record

  CommandLine cmd;
  cmd.AddValue ("nIoT", "Number of IoT nodes", nIoT);
  cmd.AddValue ("simTime", "Simulation duration (s)", simTime);
  cmd.AddValue ("attackStart", "MITM attack start time (s)", attackStart);
  cmd.AddValue ("attackStop", "MITM attack stop time (s)", attackStop);
  cmd.AddValue ("spoofIp", "Spoofed A record IP", spoofIp);
  cmd.Parse (argc, argv);

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);
  Time::SetResolution (Time::NS);

  // Nodes
  NodeContainer iotNodes; iotNodes.Create(nIoT);
  NodeContainer inlineNode; inlineNode.Create(1);
  NodeContainer gateway; gateway.Create(1);
  NodeContainer server; server.Create(1);

  // CSMA LAN (IoT + inline)
  NodeContainer csmaLan;
  csmaLan.Add(iotNodes);
  csmaLan.Add(inlineNode.Get(0));
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue ("100Mbps"));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  NetDeviceContainer csmaDevices = csma.Install (csmaLan);

  // p2p inline <-> gateway
  PointToPointHelper p2pInline;
  p2pInline.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
  p2pInline.SetChannelAttribute ("Delay", StringValue ("2ms"));
  NodeContainer inlineToGw; inlineToGw.Add(inlineNode.Get(0)); inlineToGw.Add(gateway.Get(0));
  NetDeviceContainer p2pInlineDevices = p2pInline.Install (inlineToGw);

  // p2p gateway <-> server
  PointToPointHelper p2pGwServer;
  p2pGwServer.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
  p2pGwServer.SetChannelAttribute ("Delay", StringValue ("2ms"));
  NodeContainer gwToServer; gwToServer.Add(gateway.Get(0)); gwToServer.Add(server.Get(0));
  NetDeviceContainer p2pGwServerDevices = p2pGwServer.Install (gwToServer);

  // Internet stack
  InternetStackHelper internet;
  internet.Install(iotNodes);
  internet.Install(inlineNode);
  internet.Install(gateway);
  internet.Install(server);

  // IP addressing
  Ipv4AddressHelper address;
  address.SetBase("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer csmaIf = address.Assign(csmaDevices);

  address.SetBase("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer p2pInlineIf = address.Assign(p2pInlineDevices);

  address.SetBase("10.1.3.0", "255.255.255.0");
  Ipv4InterfaceContainer p2pGwServerIf = address.Assign(p2pGwServerDevices);

    // Print some IPs for clarity
  Ptr<Ipv4> inlineIpv4 = inlineNode.Get(0)->GetObject<Ipv4>();
  Ipv4Address inlineCsmaAddr = Ipv4Address::GetZero();
  // find the CSMA interface IP on inline node (skip loopback)
  for (uint32_t iface=1; iface < inlineIpv4->GetNInterfaces(); ++iface)
  {
    for (uint32_t a=0; a < inlineIpv4->GetNAddresses(iface); ++a)
    {
      Ipv4Address addr = inlineIpv4->GetAddress(iface,a).GetLocal();
      if (addr != Ipv4Address::GetLoopback())
      {
        // detect CSMA network by checking if address starts with 10.1.1
        uint32_t addrInt = addr.Get();
        uint32_t csmaNetwork = Ipv4Address("10.1.1.0").Get();
        uint32_t mask = Ipv4Mask("255.255.255.0").Get();
        if ((addrInt & mask) == (csmaNetwork & mask))
        {
          inlineCsmaAddr = addr;
        }
      }
    }
  }
  Ptr<Ipv4> serverIpv4 = server.Get(0)->GetObject<Ipv4>();
  Ipv4Address serverAddr = serverIpv4->GetAddress(1,0).GetLocal();

  // Set global variables for callbacks
  g_attackStart = attackStart;
  g_attackStop = attackStop;
  g_spoofIp = spoofIp;
  g_serverAddr = serverAddr;

  std::cout << "Inline CSMA IP: " << inlineCsmaAddr << std::endl;
  std::cout << "Server IP: " << serverAddr << std::endl;

  // Enable IP forwarding (so traffic flows through inline->gateway->server)
  inlineNode.Get(0)->GetObject<Ipv4>()->SetAttribute("IpForward", BooleanValue(true));
  gateway.Get(0)->GetObject<Ipv4>()->SetAttribute("IpForward", BooleanValue(true));
  Ipv4GlobalRoutingHelper::PopulateRoutingTables();

  // ---------- Applications ----------
  // TCP telemetry: server PacketSink + IoT BulkSend clients (unchanged)
  PacketSinkHelper tcpSinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), tcpPort));
  ApplicationContainer sinkApps = tcpSinkHelper.Install(server.Get(0));
  sinkApps.Start(Seconds(0.0));
  sinkApps.Stop(Seconds(simTime + 1.0));

  for (uint32_t i=0; i < iotNodes.GetN(); ++i)
  {
    BulkSendHelper client("ns3::TcpSocketFactory", InetSocketAddress(serverAddr, tcpPort));
    client.SetAttribute("MaxBytes", UintegerValue(maxBytes));
    ApplicationContainer app = client.Install(iotNodes.Get(i));
    double startTime = 1.0 + i * 0.2;
    double stopTime = simTime - 1.0;
    app.Start(Seconds(startTime)); app.Stop(Seconds(stopTime));
  }

  // ---------- DNS backend on real server ----------
  // A simple DNS responder on server that replies with 192.0.2.1 for A queries (same as earlier)
  // We'll install a callback socket on the server to respond to queries
  TypeId udpTid = TypeId::LookupByName("ns3::UdpSocketFactory");
  Ptr<Socket> serverDnsSocket = Socket::CreateSocket(server.Get(0), udpTid);
  InetSocketAddress serverLocal = InetSocketAddress(Ipv4Address::GetAny(), DNS_PORT);
  serverDnsSocket->Bind(serverLocal);

  // Server DNS receive callback (simple response to A queries with 192.0.2.1)
  serverDnsSocket->SetRecvCallback(MakeCallback(&ServerDnsRecv));

  // ---------- Inline DNS resolver & MITM logic ----------
  // Inline socket bound to port 53 on inline node - receives queries from clients and responses from upstream
  Ptr<Socket> inlineDnsSock = Socket::CreateSocket(inlineNode.Get(0), udpTid);
  InetSocketAddress inlineLocal = InetSocketAddress(Ipv4Address::GetAny(), DNS_PORT);
  inlineDnsSock->Bind(inlineLocal);

  // Inline receive callback - handles both queries from clients and responses from upstream server
  inlineDnsSock->SetRecvCallback(MakeCallback(&InlineDnsRecv));

  // ---------- Install DNS clients on IoT nodes (send queries to inline resolver) ----------
  // Each IoT node will send DNS queries periodically to inlineCsmaAddr:53
  // seed std::rand for txid variety (not required; ns3 RNG is separate)
  std::srand(1);
  
  for (uint32_t i = 0; i < iotNodes.GetN(); ++i)
  {
    Ptr<Node> n = iotNodes.Get(i);
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    Ptr<Socket> sock = Socket::CreateSocket(n, tid);
    InetSocketAddress remote = InetSocketAddress(inlineCsmaAddr, DNS_PORT);
    sock->Connect(remote);

    // schedule queries every dnsInterval seconds starting at 1.5s until simTime - 1.0s
    double t = 1.5 + i * 0.1; // slight staggering per node
    uint32_t counter = 0;
    while (t < simTime - 1.0)
    {
      uint16_t txid = static_cast<uint16_t>(std::rand() & 0xFFFF);
      uint16_t qtype = GetRandomQueryType(); // Random query type: A, AAAA, MX, or TXT
      // schedule a send; capture sock, txid, and qtype
      Simulator::Schedule(Seconds(t), &SendDnsQuery, sock, txid, qtype);
      t += dnsInterval;
      ++counter;
    }
  }

  // ---------- PCAP + FlowMonitor ----------
  if (enablePcap)
  {
    csma.EnablePcapAll("iiot_csma");
    p2pInline.EnablePcapAll("iiot_p2p_inline");
    p2pGwServer.EnablePcapAll("iiot_p2p_gw_server");
  }

  Ptr<FlowMonitor> flowMonitor;
  FlowMonitorHelper flowHelper;
  if (enableFlowMonitor)
  {
    flowMonitor = flowHelper.InstallAll();
  }

  // write meta.json with attack window info
  {
    std::ofstream meta("meta_mitm.json");
    meta << "{\n";
    meta << "  \"attack\": { \"type\": \"dns_spoof\", \"start\": " << attackStart << ", \"stop\": " << attackStop << ", \"spoof_ip\": \"" << spoofIp << "\" }\n";
    meta << "}\n";
    meta.close();
  }

  Simulator::Stop(Seconds(simTime + 0.5));
  Simulator::Run();

  if (enableFlowMonitor)
  {
    flowMonitor->CheckForLostPackets();
    flowMonitor->SerializeToXmlFile("iiot_flowmon_mitm.xml", true, true);
  }

  Simulator::Destroy();
  return 0;
}

// Helper function defined after main to schedule socket sends (must be global C function)
void SendPacketTo(Ptr<Socket> sock, std::vector<uint8_t> q)
{
  if (!sock) return;
  Ptr<Packet> p = Create<Packet>(q.data(), q.size());
  sock->Send(p);
}
