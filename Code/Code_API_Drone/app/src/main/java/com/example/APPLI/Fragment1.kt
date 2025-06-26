package com.example.APPLI

import android.Manifest
import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothSocket
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ListView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import java.io.IOException
import java.io.InputStream
import java.util.UUID
import androidx.core.app.ActivityCompat
import android.content.pm.PackageManager
import android.widget.TextView

/**
 * Fragment gérant la connexion Bluetooth Low Energy (BLE),
 * l'envoi de commandes et l'affichage des données reçues.
 */
class Fragment1 : Fragment() {
    private val TAG = "Fragment1"
    private lateinit var btnPermissionsBT: Button
    private lateinit var btnActiverBT: Button
    private lateinit var btnRechercheAppareils: Button
    private lateinit var btnEffacerListe: Button
    private lateinit var listView: ListView
    private lateinit var bluetoothManager: BluetoothManager
    private lateinit var bluetoothAdapter: BluetoothAdapter
    private lateinit var deviceList: ArrayList<String>
    private lateinit var deviceAdapter: ArrayAdapter<String>
    private var isDiscovering = false
    private var connectThread: ConnectThread? = null
    private var connectedThread: ConnectedThread? = null
    private var bluetoothSocket: BluetoothSocket? = null
    private lateinit var textView: TextView
    
    // ViewModel partagé pour communiquer avec les autres fragments
    private val sharedDataManager: SharedDataManager by activityViewModels()

    private val requetePermissions: ActivityResultLauncher<Array<String>> =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            Log.d(TAG, "Permission request result received: $permissions")
            if (permissions.values.all { it }) {
                Log.d(TAG, "All permissions granted")
                btnActiverBT.isEnabled = true
                Toast.makeText(activity, "Permissions OK", Toast.LENGTH_SHORT).show()
            } else {
                Log.w(TAG, "Some permissions were denied: ${permissions.filter { !it.value }.keys}")
            }
        }

    private val requeteActivationBT =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            Log.d(TAG, "Bluetooth activation result received: ${result.resultCode}")
            if (result.resultCode == Activity.RESULT_OK) {
                Log.d(TAG, "User enabled Bluetooth")
                Toast.makeText(activity, "L'utilisateur a activé le BT",
                    Toast.LENGTH_SHORT).show()
            } else {
                Log.w(TAG, "User denied Bluetooth activation")
                Toast.makeText(activity, "L'utilisateur a refusé",
                    Toast.LENGTH_SHORT).show()
            }
        }

    // BroadcastReceiver pour la découverte d'appareils
    private val discoveryReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                BluetoothDevice.ACTION_FOUND -> {
                    val device: BluetoothDevice? = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE)
                    device?.let {
                        val deviceName = it.name ?: "Appareil inconnu"
                        val deviceAddress = it.address
                        val deviceInfo = "$deviceName\n$deviceAddress"
                        
                        if (!deviceList.contains(deviceInfo)) {
                            deviceList.add(deviceInfo)
                            deviceAdapter.notifyDataSetChanged()
                            Log.d(TAG, "Appareil trouvé: $deviceInfo")
                        }
                    }
                }
                BluetoothAdapter.ACTION_DISCOVERY_STARTED -> {
                    Log.d(TAG, "Découverte Bluetooth démarrée")
                    isDiscovering = true
                    btnRechercheAppareils.text = "Arrêter la recherche"
                    Toast.makeText(activity, "Recherche d'appareils démarrée", Toast.LENGTH_SHORT).show()
                }
                BluetoothAdapter.ACTION_DISCOVERY_FINISHED -> {
                    Log.d(TAG, "Découverte Bluetooth terminée")
                    isDiscovering = false
                    btnRechercheAppareils.text = "Recherche des appareils"
                    Toast.makeText(activity, "Recherche terminée", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    // BroadcastReceiver pour les changements d'état d'appairage
    private val bondStateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                BluetoothDevice.ACTION_BOND_STATE_CHANGED -> {
                    val device = intent.getParcelableExtra<BluetoothDevice>(BluetoothDevice.EXTRA_DEVICE)
                    val bondState = intent.getIntExtra(BluetoothDevice.EXTRA_BOND_STATE, BluetoothDevice.ERROR)
                    
                    device?.let {
                        when (bondState) {
                            BluetoothDevice.BOND_BONDED -> {
                                Log.i(TAG, "Appairage réussi avec ${it.name}")
                                Toast.makeText(activity, "Appairage réussi avec ${it.name}", Toast.LENGTH_SHORT).show()
                            }
                            BluetoothDevice.BOND_NONE -> {
                                Log.i(TAG, "Appairage échoué avec ${it.name}")
                                Toast.makeText(activity, "Appairage échoué avec ${it.name}", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                }
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        Log.d(TAG, "onCreateView called")
        val view = inflater.inflate(R.layout.fragment_1, container, false)

        // Initialisation des vues
        btnPermissionsBT = view.findViewById(R.id.btnPermissionsBT)
        btnActiverBT = view.findViewById(R.id.btnActiverBT)
        btnRechercheAppareils = view.findViewById(R.id.btnAppareilsAssocies)
        btnEffacerListe = view.findViewById(R.id.btnEffacerListe)
        listView = view.findViewById(R.id.listView1)
        textView = view.findViewById(R.id.textView)
        Log.d(TAG, "Views initialized")

        // Initialisation de la liste d'appareils
        deviceList = ArrayList()
        deviceAdapter = ArrayAdapter(requireContext(), android.R.layout.simple_list_item_1, deviceList)
        listView.adapter = deviceAdapter
        
        // Gestionnaire de clic pour la sélection d'un appareil
        listView.setOnItemClickListener { _, _, position, _ ->
            val selectedDeviceInfo = deviceList[position]
            val address: String = selectedDeviceInfo.split('\n')[1]
            val device: BluetoothDevice? = if (ActivityCompat.checkSelfPermission(requireContext(), Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED) {
                bluetoothAdapter.getRemoteDevice(address)
            } else {
                Toast.makeText(requireContext(), "Permission Bluetooth requise", Toast.LENGTH_SHORT).show()
                null
            }
            
            if (device != null) {
                Log.i(TAG, "Device choisi - ${device.name} - $address")
                
                // Vérifier si l'appareil est déjà appairé
                if (device.bondState == BluetoothDevice.BOND_BONDED) {
                    Log.i(TAG, "Appareil déjà appairé, démarrage de la connexion")
                    Toast.makeText(activity, "Connexion à ${device.name}...", Toast.LENGTH_SHORT).show()
                    connectThread = ConnectThread(device)
                    connectThread?.start()
                } else {
                    Log.i(TAG, "Appareil non appairé, création de l'appairage")
                    Toast.makeText(activity, "Appairage avec ${device.name}...", Toast.LENGTH_SHORT).show()
                    
                    // Créer l'appairage
                    device.createBond()
                    
                    // Démarrer la connexion après un délai pour permettre l'appairage
                    Handler(Looper.getMainLooper()).postDelayed({
                        if (device.bondState == BluetoothDevice.BOND_BONDED) {
                            connectThread = ConnectThread(device)
                            connectThread?.start()
                        } else {
                            Toast.makeText(activity, "Échec de l'appairage avec ${device.name}", Toast.LENGTH_LONG).show()
                        }
                    }, 3000) // Attendre 3 secondes pour l'appairage
                }
            } else {
                Log.e(TAG, "Impossible de récupérer l'appareil pour l'adresse: $address")
                Toast.makeText(activity, "Erreur: appareil non trouvé", Toast.LENGTH_SHORT).show()
            }
        }
        
        Log.d(TAG, "Device list and adapter initialized")

        // Configuration initiale
        btnActiverBT.isEnabled = false
        Log.d(TAG, "Initial configuration set")

        btnPermissionsBT.setOnClickListener {
            Log.d(TAG, "Permissions button clicked")
            lateinit var BT_PERMISSIONS: Array<String>
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                Log.d(TAG, "Using Android 12+ permissions")
                BT_PERMISSIONS = arrayOf(
                    Manifest.permission.BLUETOOTH_CONNECT,
                    Manifest.permission.BLUETOOTH_SCAN
                )
            } else {
                Log.d(TAG, "Using legacy permissions")
                BT_PERMISSIONS = arrayOf(
                    Manifest.permission.BLUETOOTH_ADMIN,
                    Manifest.permission.BLUETOOTH,
                    Manifest.permission.ACCESS_FINE_LOCATION
                )
            }
            Log.d(TAG, "Requesting permissions: ${BT_PERMISSIONS.joinToString()}")
            requetePermissions.launch(BT_PERMISSIONS)
        }

        btnActiverBT.setOnClickListener {
            Log.d(TAG, "Activate BT button clicked")
            if (bluetoothAdapter.isEnabled) {
                Log.d(TAG, "Bluetooth is already enabled")
                Toast.makeText(activity, "BT déjà activé", Toast.LENGTH_SHORT).show()
            } else {
                Log.d(TAG, "Requesting Bluetooth activation")
                val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
                requeteActivationBT.launch(enableBtIntent)
            }
        }

        btnRechercheAppareils.setOnClickListener {
            Log.d(TAG, "Search devices button clicked")
            if (!bluetoothAdapter.isEnabled) {
                Toast.makeText(activity, "Veuillez d'abord activer le Bluetooth", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            
            if (isDiscovering) {
                // Arrêter la recherche
                bluetoothAdapter.cancelDiscovery()
                Log.d(TAG, "Discovery cancelled")
            } else {
                // Démarrer la recherche
                deviceList.clear()
                deviceAdapter.notifyDataSetChanged()
                bluetoothAdapter.startDiscovery()
                Log.d(TAG, "Discovery started")
            }
        }

        btnEffacerListe.setOnClickListener {
            Log.d(TAG, "Clear list button clicked")
            deviceList.clear()
            deviceAdapter.notifyDataSetChanged()
            Toast.makeText(activity, "Liste effacée", Toast.LENGTH_SHORT).show()
        }

        // Tester si l'appareil ciblé autorise l'interface BT
        Log.d(TAG, "Initializing Bluetooth manager and adapter")
        bluetoothManager = requireContext().getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothAdapter = bluetoothManager.adapter
        if (bluetoothAdapter == null) {
            Log.e(TAG, "Device does not support Bluetooth")
            Toast.makeText(activity, "La machine ne possède pas le Bluetooth",
                Toast.LENGTH_SHORT).show()
        } else {
            Log.d(TAG, "Bluetooth adapter found and initialized")
            btnPermissionsBT.isEnabled = true
            Toast.makeText(activity, "Interface BT existe", Toast.LENGTH_SHORT).show()
        }

        textView.text = "En attente des permissions Bluetooth..."

        // --- Configuration des listeners pour les boutons ---
        setupListeners()

        // --- Observation des données partagées pour affichage ---
        observeSharedData()

        return view
    }

    override fun onResume() {
        super.onResume()
        Log.d(TAG, "onResume called")
        
        // Enregistrer le BroadcastReceiver pour la découverte
        val discoveryFilter = IntentFilter().apply {
            addAction(BluetoothDevice.ACTION_FOUND)
            addAction(BluetoothAdapter.ACTION_DISCOVERY_STARTED)
            addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED)
        }
        requireContext().registerReceiver(discoveryReceiver, discoveryFilter)
        
        // Enregistrer le BroadcastReceiver pour l'état d'appairage
        val bondFilter = IntentFilter().apply {
            addAction(BluetoothDevice.ACTION_BOND_STATE_CHANGED)
        }
        requireContext().registerReceiver(bondStateReceiver, bondFilter)
        
        Log.d(TAG, "BroadcastReceivers registered")
    }

    override fun onPause() {
        super.onPause()
        Log.d(TAG, "onPause called")
        
        // Désinscrire les BroadcastReceivers
        try {
            requireContext().unregisterReceiver(discoveryReceiver)
            requireContext().unregisterReceiver(bondStateReceiver)
            Log.d(TAG, "BroadcastReceivers unregistered")
        } catch (e: Exception) {
            Log.e(TAG, "Error unregistering receivers: ${e.message}")
        }
        
        // Arrêter la découverte si elle est en cours
        if (isDiscovering && bluetoothAdapter.isDiscovering) {
            bluetoothAdapter.cancelDiscovery()
            Log.d(TAG, "Discovery cancelled in onPause")
        }
    }

    // Thread de connexion Bluetooth
    inner class ConnectThread(private val device: BluetoothDevice) : Thread() {
        private val uuid = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
        private val mmSocket: BluetoothSocket = device.createRfcommSocketToServiceRecord(uuid)

        override fun run() {
            Log.d(TAG, "=== DÉBUT CONNECTTHREAD ===")
            Log.d(TAG, "Tentative de connexion à l'appareil: ${device.name} (${device.address})")
            
            // Cancel discovery because it otherwise slows down the connection.
            bluetoothAdapter.cancelDiscovery()
            Log.d(TAG, "Découverte Bluetooth annulée")

            try {
                Log.i(TAG, "début attente connexion")
                mmSocket.connect()
                Log.i(TAG, "fin attente connexion - connexion OK")
                
                // Stocker le socket pour une utilisation ultérieure
                bluetoothSocket = mmSocket
                Log.d(TAG, "Socket Bluetooth stocké")
                
                // Lancer le thread de réception des données
                connectedThread = ConnectedThread(mmSocket)
                connectedThread?.start()
                Log.d(TAG, "ConnectedThread démarré")
                
                // Mettre à jour l'interface utilisateur sur le thread principal
                requireActivity().runOnUiThread {
                    Toast.makeText(activity, "Connexion établie avec "+device.name, Toast.LENGTH_LONG).show()
                    textView.text = "Connecté à: ${device.name}\n${device.address}"
                    btnRechercheAppareils.text = "Déconnecter"
                    btnRechercheAppareils.setOnClickListener {
                        disconnect()
                    }
                }
                Log.d(TAG, "Interface utilisateur mise à jour")
                
            } catch (e: IOException) {
                Log.e(TAG, "Erreur de connexion", e)
                Log.e(TAG, "Détails de l'erreur: ${e.message}")
                e.printStackTrace()
                requireActivity().runOnUiThread {
                    Toast.makeText(activity, "Erreur de connexion: ${e.message}", Toast.LENGTH_LONG).show()
                    textView.text = "Erreur de connexion: ${e.message}"
                }
            }
            Log.d(TAG, "=== FIN CONNECTTHREAD ===")
        }

        fun cancel() {
            try {
                mmSocket.close()
                Log.d(TAG, "Socket Bluetooth fermé")
            } catch (e: IOException) {
                Log.e(TAG, "Erreur lors de la fermeture du socket", e)
            }
        }
    }

    // Méthode pour déconnecter
    private fun disconnect() {
        // Arrêter le thread de réception des données
        connectedThread?.cancel()
        connectedThread = null
        
        // Arrêter le thread de connexion
        connectThread?.cancel()
        connectThread = null
        
        // Fermer le socket
        try {
            bluetoothSocket?.close()
            bluetoothSocket = null
        } catch (e: IOException) {
            Log.e(TAG, "Erreur lors de la fermeture du socket", e)
        }
        
        // Restaurer l'interface utilisateur
        textView.text = "Déconnecté"
        btnRechercheAppareils.text = "Recherche des appareils"
        
        Toast.makeText(activity, "Déconnecté", Toast.LENGTH_SHORT).show()
    }

    // Thread pour gérer la réception des données Bluetooth
    inner class ConnectedThread(private val mmSocket: BluetoothSocket) : Thread() {
        private val mmInStream: InputStream = mmSocket.inputStream
        private val mmBuffer: ByteArray = ByteArray(1024) // mmBuffer store for the stream

        override fun run() {
            Log.d(TAG, "ConnectedThread démarré")
            var numBytes: Int // bytes returned from read()
            
            // Keep listening to the InputStream until an exception occurs.
            while (true) {
                // Read from the InputStream.
                numBytes = try {
                    Log.d(TAG, "Tentative de lecture depuis l'InputStream...")
                    mmInStream.read(mmBuffer)
                } catch (e: IOException) {
                    Log.d(TAG, "Input stream was disconnected", e)
                    break
                }
                
                Log.d(TAG, "Bytes lus: $numBytes")
                Log.i(TAG, "Données reçues (bytes): ${mmBuffer.contentToString()}")
                
                // Convertir les bytes reçus en String
                if (numBytes > 0) {
                    val receivedData = String(mmBuffer, 0, numBytes)
                    Log.i(TAG, "Données reçues (String): '$receivedData'")
                    
                    // Parser les données et les envoyer aux autres fragments
                    parseAndDistributeData(receivedData)
                    
                    // La mise à jour du TextView avec les données reçues est supprimée.
                } else {
                    Log.d(TAG, "Aucun byte lu (numBytes = $numBytes)")
                }
            }
            Log.d(TAG, "ConnectedThread terminé")
        }

        fun cancel() {
            try {
                mmInStream.close()
                Log.d(TAG, "InputStream fermé")
            } catch (e: IOException) {
                Log.e(TAG, "Erreur lors de la fermeture de l'InputStream", e)
            }
        }
    }

    // Méthode pour parser et distribuer les données aux fragments
    private fun parseAndDistributeData(data: String) {
        Log.d(TAG, "Parsing des données: '$data'")
        
        val parts = data.split('|')
        
        // La première partie est pour le GPS
        if (parts.isNotEmpty()) {
            val gpsData = parts[0]
            sharedDataManager.updateGpsData(gpsData)
            Log.d(TAG, "Données GPS envoyées: $gpsData")
        }

        // La deuxième partie (si elle existe) est pour le son
        if (parts.size > 1) {
            val soundData = parts[1]
            sharedDataManager.updateSoundData(soundData)
            Log.d(TAG, "Données sonores envoyées: $soundData")
        }
    }

    private fun checkConnection() {
        Log.d(TAG, "=== VÉRIFICATION DE LA CONNEXION ===")
        
        // Vérifier l'état du Bluetooth
        val bluetoothEnabled = bluetoothAdapter.isEnabled
        Log.d(TAG, "Bluetooth activé: $bluetoothEnabled")
        
        // Vérifier l'état du socket
        val socketExists = bluetoothSocket != null
        Log.d(TAG, "Socket Bluetooth existe: $socketExists")
        
        // Vérifier l'état du thread de connexion
        val connectThreadRunning = connectThread?.isAlive ?: false
        Log.d(TAG, "ConnectThread en cours: $connectThreadRunning")
        
        // Vérifier l'état du thread de réception
        val connectedThreadRunning = connectedThread?.isAlive ?: false
        Log.d(TAG, "ConnectedThread en cours: $connectedThreadRunning")
        
        // Vérifier si le socket est connecté
        val isConnected = try {
            bluetoothSocket?.isConnected ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Erreur lors de la vérification de la connexion: ${e.message}")
            false
        }
        Log.d(TAG, "Socket connecté: $isConnected")
        
        // Afficher un résumé
        val status = StringBuilder()
        status.append("État Bluetooth: ${if (bluetoothEnabled) "OK" else "DÉSACTIVÉ"}\n")
        status.append("Socket: ${if (socketExists) "Créé" else "Non créé"}\n")
        status.append("Thread connexion: ${if (connectThreadRunning) "En cours" else "Arrêté"}\n")
        status.append("Thread réception: ${if (connectedThreadRunning) "En cours" else "Arrêté"}\n")
        status.append("Connexion: ${if (isConnected) "ÉTABLIE" else "NON ÉTABLIE"}")
        
        Log.d(TAG, "Résumé de la connexion:\n$status")
        
        // Afficher dans l'interface
        textView.text = "État de la connexion:\n\n$status"
        
        Toast.makeText(activity, "Vérification terminée", Toast.LENGTH_SHORT).show()
        
        Log.d(TAG, "=== FIN VÉRIFICATION DE LA CONNEXION ===")
    }

    /**
     * Configure les actions des boutons (Scan, Connexion, Déconnexion, Envoi de données).
     */
    private fun setupListeners() {
        // ... existing code ...
    }

    /**
     * Observe les données partagées (GPS, Son) et les affiche dans les TextViews.
     */
    private fun observeSharedData() {
        // ... existing code ...
    }

    /**
     * Scanne les appareils BLE à proximité.
     */
    private fun scanLeDevice() {
        // ... existing code ...
    }

    /**
     * Se connecte à un appareil BLE sélectionné.
     */
    private fun connectToDevice(device: BluetoothDevice) {
        // ... existing code ...
    }

    /**
     * Se déconnecte de l'appareil BLE.
     */
    private fun disconnectFromDevice() {
        // ... existing code ...
    }

    /**
     * Envoie des données à l'appareil BLE connecté.
     */
    private fun sendData(data: String) {
        // ... existing code ...
    }

    /**
     * Récepteur pour les mises à jour de l'état de la connexion et les données reçues.
     */
    private val gattUpdateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            // ... existing code ...
        }
    }

    /**
     * Gère les demandes de permissions Bluetooth et localisation.
     */
    private fun checkAndRequestPermissions() {
        // ... existing code ...
    }

    companion object {
        // ... Constantes et instance du fragment ...
    }
} 