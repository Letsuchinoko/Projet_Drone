package com.example.APPLI

import android.Manifest
import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothSocket
import android.content.*
import android.content.pm.PackageManager
import android.location.LocationManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.*
import android.widget.*
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import java.io.IOException
import java.util.*
import kotlin.concurrent.thread

class Fragment1 : Fragment() {
    companion object {
        private const val TAG = "BluetoothDebug"
        private const val UUID_STRING = "00001101-0000-1000-8000-00805F9B34FB" // UUID standard pour SPP
        private const val REQUEST_ENABLE_LOCATION = 1001
    }

    // Variables d'interface
    private lateinit var textView: TextView
    private lateinit var btnPermissionsBT: Button
    private lateinit var btnActiverBT: Button
    private lateinit var btnRechercher: Button
    private lateinit var btnEffacerListe: Button
    private lateinit var listView: ListView
    private lateinit var arrayAdapter: ArrayAdapter<String>
    private val listeAppareils = mutableListOf<String>()
    private val appareilsMap = mutableMapOf<String, BluetoothDevice>()

    // Variables Bluetooth
    private lateinit var bluetoothManager: BluetoothManager
    private lateinit var bluetoothAdapter: BluetoothAdapter
    private var bluetoothSocket: BluetoothSocket? = null
    private var deviceConnected: BluetoothDevice? = null

    // Receiver pour les événements Bluetooth
    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                BluetoothDevice.ACTION_FOUND -> {
                    Log.d(TAG, "Appareil trouvé")
                    val device: BluetoothDevice? = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE)
                    device?.let { 
                        val name = it.name ?: "Inconnu"
                        val address = it.address
                        val rssi = intent.getShortExtra(BluetoothDevice.EXTRA_RSSI, Short.MIN_VALUE)
                        Log.d(TAG, "Appareil trouvé: $name ($address) RSSI: $rssi")
                        ajouterAppareil(it)
                    }
                }
                BluetoothAdapter.ACTION_DISCOVERY_FINISHED -> {
                    Log.d(TAG, "Recherche terminée")
                    Toast.makeText(activity, "Recherche terminée", Toast.LENGTH_SHORT).show()
                }
                BluetoothAdapter.ACTION_DISCOVERY_STARTED -> {
                    Log.d(TAG, "Recherche démarrée")
                }
            }
        }
    }

    // Gestionnaire des permissions
    private val requetePermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.entries.all { it.value }
        Log.d(TAG, "Résultat des permissions: $permissions")
        if (allGranted) {
            activerBoutons()
            Toast.makeText(activity, "Permissions accordées", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(activity, "Permissions refusées", Toast.LENGTH_SHORT).show()
        }
    }

    // Gestionnaire d'activation Bluetooth
    private val requeteActivationBT = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            Log.d(TAG, "Bluetooth activé par l'utilisateur")
            activerBoutons()
            Toast.makeText(activity, "Bluetooth activé", Toast.LENGTH_SHORT).show()
        } else {
            Log.d(TAG, "Bluetooth refusé par l'utilisateur")
            Toast.makeText(activity, "Bluetooth refusé", Toast.LENGTH_SHORT).show()
        }
    }

    // Gestionnaire d'activation de la localisation
    private val requeteActivationLocation = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            Log.d(TAG, "Localisation activée")
            Toast.makeText(activity, "Localisation activée", Toast.LENGTH_SHORT).show()
            rechercherAppareils()
        } else {
            Log.d(TAG, "Localisation refusée")
            Toast.makeText(activity, "La localisation est nécessaire pour la recherche Bluetooth", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_1, container, false)
        initialiserVues(view)
        initialiserBluetooth()
        configurerBoutons()
        return view
    }

    private fun initialiserVues(view: View) {
        textView = view.findViewById(R.id.textView)
        btnPermissionsBT = view.findViewById(R.id.btnPermissionsBT)
        btnActiverBT = view.findViewById(R.id.btnActiverBT)
        btnRechercher = view.findViewById(R.id.btnRechercher)
        btnEffacerListe = view.findViewById(R.id.btnEffacerListe)
        listView = view.findViewById(R.id.listView1)

        arrayAdapter = ArrayAdapter(requireContext(), android.R.layout.simple_list_item_1, listeAppareils)
        listView.adapter = arrayAdapter

        // Ajouter le listener pour les clics sur les éléments de la liste
        listView.setOnItemClickListener { _, _, position, _ ->
            val selectedItem = listeAppareils[position]
            val address = selectedItem.split("\n")[1]
            Log.d(TAG, "Tentative de connexion à l'appareil: $address")
            appareilsMap[address]?.let { device ->
                if (deviceConnected?.address == device.address) {
                    deconnecterAppareil()
                } else {
                    connecterAppareil(device)
                }
            }
        }

        textView.text = "En attente des permissions Bluetooth..."
        desactiverBoutons()
    }

    private fun initialiserBluetooth() {
        try {
            bluetoothManager = requireContext().getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
            bluetoothAdapter = bluetoothManager.adapter
            if (bluetoothAdapter == null) {
                Log.e(TAG, "Bluetooth non disponible sur cet appareil")
                Toast.makeText(activity, "Bluetooth non disponible", Toast.LENGTH_SHORT).show()
            } else {
                Log.d(TAG, "Bluetooth disponible")
                btnPermissionsBT.isEnabled = true
                if (bluetoothAdapter.isEnabled) {
                    activerBoutons()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Erreur lors de l'initialisation du Bluetooth: ${e.message}")
            Toast.makeText(activity, "Erreur Bluetooth: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun configurerBoutons() {
        btnPermissionsBT.setOnClickListener { demanderPermissions() }
        btnActiverBT.setOnClickListener { activerBluetooth() }
        btnRechercher.setOnClickListener { rechercherAppareils() }
        btnEffacerListe.setOnClickListener { effacerListe() }
    }

    private fun demanderPermissions() {
        val permissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(
                Manifest.permission.BLUETOOTH_CONNECT,
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        } else {
            arrayOf(
                Manifest.permission.BLUETOOTH_ADMIN,
                Manifest.permission.BLUETOOTH,
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        }
        Log.d(TAG, "Demande des permissions: ${permissions.joinToString()}")
        requetePermissions.launch(permissions)
    }

    private fun activerBluetooth() {
        if (!bluetoothAdapter.isEnabled) {
            Log.d(TAG, "Demande d'activation du Bluetooth")
            val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
            requeteActivationBT.launch(enableBtIntent)
        } else {
            Log.d(TAG, "Bluetooth déjà activé")
            Toast.makeText(activity, "Bluetooth déjà activé", Toast.LENGTH_SHORT).show()
        }
    }

    private fun verifierPermissions(): Boolean {
        val permissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(
                Manifest.permission.BLUETOOTH_CONNECT,
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        } else {
            arrayOf(
                Manifest.permission.BLUETOOTH_ADMIN,
                Manifest.permission.BLUETOOTH,
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        }

        val permissionsManquantes = permissions.filter {
            ContextCompat.checkSelfPermission(requireContext(), it) != PackageManager.PERMISSION_GRANTED
        }

        if (permissionsManquantes.isNotEmpty()) {
            Log.e(TAG, "Permissions manquantes: ${permissionsManquantes.joinToString()}")
            Toast.makeText(activity, "Permissions manquantes", Toast.LENGTH_SHORT).show()
            return false
        }
        return true
    }

    private fun activerLocalisation() {
        try {
            val locationManager = requireContext().getSystemService(Context.LOCATION_SERVICE) as LocationManager
            if (!locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                Log.d(TAG, "Demande d'activation de la localisation")
                val intent = Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS)
                requeteActivationLocation.launch(intent)
            } else {
                Log.d(TAG, "Localisation déjà activée")
                rechercherAppareils()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Erreur lors de l'activation de la localisation: ${e.message}")
            Toast.makeText(activity, "Erreur lors de l'activation de la localisation", Toast.LENGTH_SHORT).show()
        }
    }

    private fun rechercherAppareils() {
        Log.d(TAG, "Début de la recherche d'appareils")
        
        if (!bluetoothAdapter.isEnabled) {
            Log.e(TAG, "Bluetooth non activé")
            Toast.makeText(activity, "Veuillez activer le Bluetooth", Toast.LENGTH_SHORT).show()
            return
        }

        if (!verifierPermissions()) {
            Log.e(TAG, "Permissions manquantes")
            return
        }

        // Vérifier si la localisation est activée
        val locationManager = requireContext().getSystemService(Context.LOCATION_SERVICE) as LocationManager
        if (!locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            Log.d(TAG, "Localisation non activée, demande d'activation")
            activerLocalisation()
            return
        }

        try {
            if (bluetoothAdapter.isDiscovering) {
                Log.d(TAG, "Annulation de la recherche en cours")
                bluetoothAdapter.cancelDiscovery()
            }

            listeAppareils.clear()
            appareilsMap.clear()
            arrayAdapter.notifyDataSetChanged()

            try {
                requireContext().unregisterReceiver(receiver)
            } catch (e: Exception) {
                Log.d(TAG, "Erreur lors du désenregistrement du receiver: ${e.message}")
            }

            val filter = IntentFilter().apply {
                addAction(BluetoothDevice.ACTION_FOUND)
                addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED)
                addAction(BluetoothAdapter.ACTION_DISCOVERY_STARTED)
            }
            requireContext().registerReceiver(receiver, filter)

            val started = bluetoothAdapter.startDiscovery()
            Log.d(TAG, "Résultat de startDiscovery: $started")
            
            if (started) {
                Toast.makeText(activity, "Recherche en cours...", Toast.LENGTH_SHORT).show()
            } else {
                Log.e(TAG, "Échec du démarrage de la recherche")
                Toast.makeText(activity, "Impossible de lancer la recherche", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Erreur lors de la recherche: ${e.message}")
            Toast.makeText(activity, "Erreur: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun ajouterAppareil(device: BluetoothDevice) {
        val name = device.name ?: "Inconnu"
        val address = device.address
        val entry = "$name\n$address"
        if (!listeAppareils.contains(entry)) {
            listeAppareils.add(entry)
            appareilsMap[address] = device
            arrayAdapter.notifyDataSetChanged()
        }
    }

    private inner class ConnectThread(private val device: BluetoothDevice) : Thread() {
        private var mmSocket: BluetoothSocket? = null

        init {
            try {
                // Essayer d'abord avec la méthode standard
                mmSocket = device.createRfcommSocketToServiceRecord(UUID.fromString("00001101-0000-1000-8000-00805F9B34FB"))
            } catch (e: Exception) {
                try {
                    // Si ça échoue, essayer avec la méthode alternative
                    val method = device.javaClass.getMethod("createRfcommSocket", Int::class.java)
                    mmSocket = method.invoke(device, 1) as BluetoothSocket
                } catch (e2: Exception) {
                    try {
                        // Dernier essai avec un autre UUID
                        mmSocket = device.createRfcommSocketToServiceRecord(UUID.fromString("00001101-0000-1000-8000-00805F9B34FB"))
                    } catch (e3: Exception) {
                        Log.e(TAG, "Toutes les méthodes de création de socket ont échoué")
                    }
                }
            }
        }

        override fun run() {
            bluetoothAdapter.cancelDiscovery()

            try {
                if (mmSocket == null) {
                    throw IOException("Socket non créé")
                }

                // Vérifier si l'appareil est déjà connecté
                if (device.bondState == BluetoothDevice.BOND_BONDED) {
                    Log.d(TAG, "Appareil déjà appairé")
                }

                mmSocket?.connect()
                
                activity?.runOnUiThread {
                    deviceConnected = device
                    textView.text = "Connecté à ${device.name}"
                    Toast.makeText(activity, "Connecté", Toast.LENGTH_SHORT).show()
                }
                
                bluetoothSocket = mmSocket
            } catch (e: IOException) {
                Log.e(TAG, "Erreur connexion: ${e.message}")
                activity?.runOnUiThread {
                    Toast.makeText(activity, "Échec connexion", Toast.LENGTH_SHORT).show()
                    textView.text = "Échec connexion"
                }
                cancel()
            }
        }

        fun cancel() {
            try {
                mmSocket?.close()
            } catch (e: IOException) {
                Log.e(TAG, "Erreur fermeture socket: ${e.message}")
            }
        }
    }

    private fun connecterAppareil(device: BluetoothDevice) {
        if (!verifierPermissions()) {
            return
        }

        if (!bluetoothAdapter.isEnabled) {
            Toast.makeText(activity, "Bluetooth non activé", Toast.LENGTH_SHORT).show()
            return
        }

        // Vérifier si l'appareil est déjà appairé
        if (device.bondState != BluetoothDevice.BOND_BONDED) {
            Log.d(TAG, "Appareil non appairé, tentative d'appairage")
            device.createBond()
        }

        Log.d(TAG, "Connexion à ${device.name}")
        Toast.makeText(activity, "Connexion...", Toast.LENGTH_SHORT).show()
        ConnectThread(device).start()
    }

    private fun deconnecterAppareil() {
        try {
            Log.d(TAG, "Déconnexion de l'appareil")
            bluetoothSocket?.close()
            deviceConnected = null
            textView.text = "Déconnecté"
            Toast.makeText(activity, "Déconnecté", Toast.LENGTH_SHORT).show()
        } catch (e: IOException) {
            Log.e(TAG, "Erreur lors de la déconnexion: ${e.message}")
            Toast.makeText(activity, "Erreur lors de la déconnexion", Toast.LENGTH_SHORT).show()
        }
    }

    private fun effacerListe() {
        listeAppareils.clear()
        appareilsMap.clear()
        arrayAdapter.notifyDataSetChanged()
        Toast.makeText(activity, "Liste effacée", Toast.LENGTH_SHORT).show()
    }

    private fun activerBoutons() {
        btnActiverBT.isEnabled = true
        btnRechercher.isEnabled = true
    }

    private fun desactiverBoutons() {
        btnActiverBT.isEnabled = false
        btnRechercher.isEnabled = false
    }

    override fun onDestroyView() {
        super.onDestroyView()
        try {
            bluetoothSocket?.close()
            requireContext().unregisterReceiver(receiver)
        } catch (_: Exception) {}
    }
}
