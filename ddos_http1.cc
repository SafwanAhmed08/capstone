#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/csma-module.h"
#include "ns3/applications-module.h"

#include <map>
#include <string>
#include <vector>
#include <random>
#include <sstream>
#include <algorithm>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("DdosHttpImprovedFixedSim");

// --------------------- SimpleHttpServer ---------------------
class SimpleHttpServer : public Application
{
public:
  SimpleHttpServer ();
  virtual ~SimpleHttpServer ();
  void Setup (uint16_t port, std::vector<uint32_t> bodySizes = std::vector<uint32_t>());

private:
  virtual void StartApplication (void);
  virtual void StopApplication  (void);

  void HandleAccept (Ptr<Socket> s, const Address& from);
  void HandleRead   (Ptr<Socket> socket);
  void OnClose      (Ptr<Socket> socket);

  Ptr<Socket> m_listenSocket;
  std::map<Ptr<Socket>, std::string> m_buffer;
  uint16_t m_port;
  std::vector<uint32_t> m_bodySizes;
  std::mt19937_64 m_rng;
  std::vector<std::pair<uint16_t, std::string>> m_responses;
};

SimpleHttpServer::SimpleHttpServer ()
  : m_listenSocket (0), m_port (80)
{
  std::random_device rd;
  m_rng.seed(rd());
  m_responses = {
    {200, "OK"},
    {301, "Moved Permanently"},
    {400, "Bad Request"},
    {404, "Not Found"},
    {500, "Internal Server Error"}
  };
}

SimpleHttpServer::~SimpleHttpServer()
{
  m_listenSocket = 0;
}

void SimpleHttpServer::Setup (uint16_t port, std::vector<uint32_t> bodySizes)
{
  m_port = port;
  m_bodySizes = bodySizes;
}

void SimpleHttpServer::StartApplication()
{
  if (m_listenSocket == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::TcpSocketFactory");
      m_listenSocket = Socket::CreateSocket (GetNode(), tid);
      InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny(), m_port);
      m_listenSocket->Bind (local);
      m_listenSocket->Listen ();
    }

  m_listenSocket->SetAcceptCallback(
    MakeNullCallback<bool, Ptr<Socket>, const Address &>(),
    MakeCallback(&SimpleHttpServer::HandleAccept, this));
}

void SimpleHttpServer::StopApplication()
{
  if (m_listenSocket != 0)
    {
      m_listenSocket->Close();
    }
}

void SimpleHttpServer::HandleAccept(Ptr<Socket> socket, const Address& from)
{
  socket->SetRecvCallback(MakeCallback(&SimpleHttpServer::HandleRead, this));
}

void SimpleHttpServer::HandleRead(Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  Address from;
  while ((packet = socket->RecvFrom(from)))
    {
      uint32_t psize = packet->GetSize();
      if (psize == 0) break;
      std::vector<uint8_t> tmp(psize);
      packet->CopyData(&tmp[0], psize);
      m_buffer[socket].append((char*)&tmp[0], psize);

      if (m_buffer[socket].find("\r\n\r\n") != std::string::npos)
        {
          std::uniform_int_distribution<size_t> resp_dist(0, m_responses.size()-1);
          auto response = m_responses[resp_dist(m_rng)];
          
          std::uniform_int_distribution<size_t> dist(0, m_bodySizes.size()-1);
          uint32_t bodyLen = m_bodySizes[dist(m_rng)];
          std::string body;
          body.resize(bodyLen ? std::min<uint32_t>(bodyLen, 20000) : 0, 'A');

          std::ostringstream oss;
          oss << "HTTP/1.1 " << response.first << " " << response.second << "\r\n";
          oss << "Server: ns3-http-server/1.0\r\n";
          oss << "Date: " << Simulator::Now().GetSeconds() << "\r\n";
          oss << "Content-Type: text/plain\r\n";
          oss << "Content-Length: " << body.size() << "\r\n";
          oss << "Connection: keep-alive\r\n\r\n";
          oss << body;

          std::string resp = oss.str();
          Ptr<Packet> rp = Create<Packet>((uint8_t*)resp.data(), resp.size());
          socket->Send(rp);
          
          Simulator::Schedule(Seconds(0.1), &Socket::Close, socket);
          m_buffer.erase(socket);
        }
    }
}

void SimpleHttpServer::OnClose(Ptr<Socket> socket)
{
  m_buffer.erase(socket);
}

// --------------------- NormalHttpClient ---------------------
class NormalHttpClient : public Application 
{
public:
  NormalHttpClient();
  virtual ~NormalHttpClient();
  void Setup(Address serverAddr, uint16_t port, Time repeatInterval);

private:
  virtual void StartApplication(void);
  virtual void StopApplication(void);
  void SendGet();
  void HandleRead(Ptr<Socket> socket);
  void ConnectionSucceeded(Ptr<Socket> socket);
  void ConnectionFailed(Ptr<Socket> socket);

  Address m_server;
  uint16_t m_port;
  Time m_interval;
  EventId m_event;
  Ptr<Socket> m_socket;
  bool m_connected;
  std::vector<std::string> m_paths;
};

NormalHttpClient::NormalHttpClient()
  : m_socket(0), m_connected(false)
{
  m_paths = {"/index.html", "/about", "/contact", "/api/data", "/login"};
}

NormalHttpClient::~NormalHttpClient()
{
  m_socket = 0;
}

void NormalHttpClient::Setup(Address serverAddr, uint16_t port, Time repeatInterval)
{
  m_server = serverAddr;
  m_port = port;
  m_interval = repeatInterval;
}

void NormalHttpClient::StartApplication()
{
  SendGet();
}

void NormalHttpClient::StopApplication()
{
  if (m_socket != 0)
    {
      m_socket->Close();
    }
  Simulator::Cancel(m_event);
}

void NormalHttpClient::ConnectionSucceeded(Ptr<Socket> socket)
{
  m_connected = true;
}

void NormalHttpClient::ConnectionFailed(Ptr<Socket> socket)
{
  m_connected = false;
}

void NormalHttpClient::HandleRead(Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  Address from;
  while ((packet = socket->RecvFrom(from)))
    {
      if (packet->GetSize() == 0) break;
    }
}

void NormalHttpClient::SendGet()
{
  if (!m_socket || !m_connected) 
    {
      TypeId tid = TypeId::LookupByName("ns3::TcpSocketFactory");
      m_socket = Socket::CreateSocket(GetNode(), tid);
      m_socket->Bind();
      m_socket->Connect(InetSocketAddress(Ipv4Address::ConvertFrom(m_server), m_port));
      m_socket->SetConnectCallback(
        MakeCallback(&NormalHttpClient::ConnectionSucceeded, this),
        MakeCallback(&NormalHttpClient::ConnectionFailed, this));
      m_socket->SetRecvCallback(MakeCallback(&NormalHttpClient::HandleRead, this));
    }

  std::string path = m_paths[rand() % m_paths.size()];
  
  std::ostringstream oss;
  oss << "GET " << path << " HTTP/1.1\r\n";
  oss << "Host: server\r\n";
  oss << "User-Agent: ns3-http-client/1.0\r\n";
  oss << "Accept: text/plain\r\n";
  oss << "Connection: keep-alive\r\n\r\n";

  std::string req = oss.str();
  Ptr<Packet> p = Create<Packet>((uint8_t*)req.data(), req.size());
  m_socket->Send(p);

  Time nextInterval = Seconds(0.1 + (rand() % 10) / 10.0);
  m_event = Simulator::Schedule(nextInterval, &NormalHttpClient::SendGet, this);
}

int main(int argc, char *argv[])
{
  bool pcap = true;
  uint32_t nServers = 1;
  uint32_t nNormalClients = 20;
  uint32_t serverPort = 80;
  double simTime = 300.0;

  CommandLine cmd;
  cmd.AddValue("pcap", "Enable PCAP tracing", pcap);
  cmd.AddValue("nServers", "Number of servers", nServers);
  cmd.AddValue("nNormalClients", "Number of normal clients", nNormalClients);
  cmd.AddValue("simTime", "Simulation time (seconds)", simTime);
  cmd.Parse(argc, argv);

  NodeContainer servers;
  servers.Create(nServers);
  
  NodeContainer normalClients;
  normalClients.Create(nNormalClients);

  CsmaHelper csma;
  csma.SetChannelAttribute("DataRate", StringValue("100Mbps"));
  csma.SetChannelAttribute("Delay", TimeValue(NanoSeconds(6560)));

  NodeContainer csmaNodes;
  csmaNodes.Add(servers);
  csmaNodes.Add(normalClients);
  
  NetDeviceContainer csmaDevices = csma.Install(csmaNodes);

  InternetStackHelper internet;
  internet.Install(csmaNodes);

  Ipv4AddressHelper ipv4;
  ipv4.SetBase("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = ipv4.Assign(csmaDevices);

  Packet::EnablePrinting();
  Config::SetDefault("ns3::PcapFileWrapper::NanosecMode", BooleanValue(true));

  std::vector<uint32_t> serverBodySizes = {10, 100, 500, 1000, 5000, 10000};
  for (uint32_t i = 0; i < nServers; ++i)
    {
      Ptr<SimpleHttpServer> serverApp = CreateObject<SimpleHttpServer>();
      serverApp->Setup(serverPort, serverBodySizes);
      servers.Get(i)->AddApplication(serverApp);
      serverApp->SetStartTime(Seconds(0.1));
      serverApp->SetStopTime(Seconds(simTime - 0.1));
    }

  for (uint32_t i = 0; i < nNormalClients; ++i)
    {
      Ptr<NormalHttpClient> client = CreateObject<NormalHttpClient>();
      double startTime = 1.0 + (rand() % 10) / 10.0;
      client->Setup(interfaces.GetAddress(0), serverPort, Seconds(0.1));
      normalClients.Get(i)->AddApplication(client);
      client->SetStartTime(Seconds(startTime));
      client->SetStopTime(Seconds(simTime - 0.1));
    }

  if (pcap)
    {
      csma.EnablePcap("ddos_http_improved_fixed-server",
                      servers.Get(0)->GetDevice(0),
                      true,
                      true);
    }

  Simulator::Stop(Seconds(simTime));
  Simulator::Run();
  Simulator::Destroy();
  return 0;
}