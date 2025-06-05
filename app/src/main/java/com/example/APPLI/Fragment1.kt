package com.example.APPLI

import android.Manifest
import android.app.Activity
import android.bluetooth.*
import android.content.*
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.*
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import java.io.IOException
import java.io.InputStream
import java.util.*

class Fragment1 : Fragment() {
    companion object {
        private const val TAG = "BluetoothDebug"
        private const val UUID_STRING = "00001101-0000-1000-8000-00805F9B34FB"
    }

    private lateinit var textView: TextView
    private lateinit var btnPermissionsBT: Button
    private lateinit var btnActiverBT: Button
    private lateinit var btnRechercher: Button
    private lateinit var btnEffacerListe: Button
    private lateinit var listView: ListView
    private lateinit var arrayAdapter: ArrayAdapter<String>
    private val listeAppareils = mutableListOf<String>()
    private val appareilsMap = mutableMapOf<String, BluetoothDevice>()

    private lateinit var bluetoothManager: BluetoothManager
    private lateinit var bluetoothAdapter: BluetoothAdapter
    private var bluetoothSocket: BluetoothSocket? = null
    private var deviceConnected: BluetoothDevice? = null

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                BluetoothDevice.ACTION_FOUND -> {
                    val device: BluetoothDevice? = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE)
                    device?.let { ajouterAppareil(it) }
                }
                BluetoothAdapter.ACTION_DISCOVERY_FINISHED -> {
                    Toast.makeText(activity, "Recherche terminée", Toast.LENGTH_SHORT).show()
                }
                BluetoothAdapter.ACTION_DISCOVERY_STARTED -> {
                    Log.d(TAG, "Recherche démarrée")
                }
            }
        }
    }

    private val bondReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (BluetoothDevice.ACTION_BOND_STATE_CHANGED == intent?.action) {
                val device = intent.getParcelableExtra<BluetoothDevice>(BluetoothDevice.EXTRA_DEVICE)
                val state = intent.getIntExtra(BluetoothDevice.EXTRA_BOND_STATE, BluetoothDevice.ERROR)
                if (state == BluetoothDevice.BOND_BONDED && device != null) {
                    Toast.makeText(requireContext(), "Appairé à ${device.name}", Toast.LENGTH_SHORT).show()
                    connecterAppareil(device)
                }
            }
        }
    }

    private val requetePermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.entries.all { it.value }) {
            activerBoutons()
            Toast.makeText(activity, "Permissions accordées", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(activity, "Permissions refusées", Toast.LENGTH_SHORT).show()
        }
    }

    private val requeteActivationBT = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        if (it.resultCode == Activity.RESULT_OK) {
            activerBoutons()
            Toast.makeText(activity, "Bluetooth activé", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(activity, "Bluetooth refusé", Toast.LENGTH_SHORT).show()
        }
    }

    private val requeteActivationLocation = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        if (it.resultCode == Activity.RESULT_OK) {
            rechercherAppareils()
        } else {
            Toast.makeText(activity, "La localisation est nécessaire", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        val view = inflater.inflate(R.layout.fragment_1, container, false)
        initialiserVues(view)
        initialiserBluetooth()
        configurerBoutons()

        requireContext().registerReceiver(bondReceiver, IntentFilter(BluetoothDevice.ACTION_BOND_STATE_CHANGED))
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

        listView.setOnItemClickListener { _, _, position, _ ->
            val selectedItem = listeAppareils[position]
            val address = selectedItem.split("\n")[1]
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
        bluetoothManager = requireContext().getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothAdapter = bluetoothManager.adapter
        if (bluetoothAdapter.isEnabled) activerBoutons()
        else btnPermissionsBT.isEnabled = true
    }

    private fun configurerBoutons() {
        btnPermissionsBT.setOnClickListener { demanderPermissions() }
        btnActiverBT.setOnClickListener { activerBluetooth() }
        btnRechercher.setOnClickListener { rechercherAppareils() }
        btnEffacerListe.setOnClickListener { effacerListe() }
    }

    private fun demanderPermissions() {
        val permissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.ACCESS_FINE_LOCATION)
        } else {
            arrayOf(Manifest.permission.BLUETOOTH, Manifest.permission.BLUETOOTH_ADMIN, Manifest.permission.ACCESS_FINE_LOCATION)
        }
        requetePermissions.launch(permissions)
    }

    private fun activerBluetooth() {
        if (!bluetoothAdapter.isEnabled) {
            val intent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
            requeteActivationBT.launch(intent)
        } else {
            Toast.makeText(activity, "Bluetooth déjà activé", Toast.LENGTH_SHORT).show()
        }
    }

    private fun verifierPermissions(): Boolean {
        val permissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.ACCESS_FINE_LOCATION)
        } else {
            arrayOf(Manifest.permission.BLUETOOTH, Manifest.permission.BLUETOOTH_ADMIN, Manifest.permission.ACCESS_FINE_LOCATION)
        }
        return permissions.all {
            ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun activerLocalisation() {
        val locationManager = requireContext().getSystemService(Context.LOCATION_SERVICE) as LocationManager
        if (!locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            val intent = Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS)
            requeteActivationLocation.launch(intent)
        } else {
            rechercherAppareils()
        }
    }

    private fun rechercherAppareils() {
        if (!bluetoothAdapter.isEnabled || !verifierPermissions()) {
            Toast.makeText(activity, "Bluetooth ou permissions manquants", Toast.LENGTH_SHORT).show()
            return
        }

        val locationManager = requireContext().getSystemService(Context.LOCATION_SERVICE) as LocationManager
        if (!locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            activerLocalisation()
            return
        }

        if (bluetoothAdapter.isDiscovering) bluetoothAdapter.cancelDiscovery()

        listeAppareils.clear()
        appareilsMap.clear()
        arrayAdapter.notifyDataSetChanged()

        try {
            requireContext().unregisterReceiver(receiver)
        } catch (_: Exception) {}

        val filter = IntentFilter().apply {
            addAction(BluetoothDevice.ACTION_FOUND)
            addAction(BluetoothAdapter.ACTION_DISCOVERY_STARTED)
            addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED)
        }
        requireContext().registerReceiver(receiver, filter)

        if (bluetoothAdapter.startDiscovery()) {
            Toast.makeText(activity, "Recherche en cours...", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(activity, "Erreur lors de la recherche", Toast.LENGTH_SHORT).show()
        }
    }

    private fun ajouterAppareil(device: BluetoothDevice) {
        val name = device.name ?: "Inconnu"
        val entry = "$name\n${device.address}"
        if (!listeAppareils.contains(entry)) {
            listeAppareils.add(entry)
            appareilsMap[device.address] = device
            arrayAdapter.notifyDataSetChanged()
        }
    }

    private inner class ConnectThread(private val device: BluetoothDevice) : Thread() {
        override fun run() {
            bluetoothAdapter.cancelDiscovery()
            try {
                val socket = device.createInsecureRfcommSocketToServiceRecord(UUID.fromString(UUID_STRING))
                bluetoothSocket = socket
                socket.connect()
                deviceConnected = device
                textView.post { textView.text = "Connecté à ${device.name}" }
                ConnectedThread(socket).start()
            } catch (e: IOException) {
                Log.e(TAG, "Erreur de connexion : ${e.message}")
                bluetoothSocket?.close()
                bluetoothSocket = null
                textView.post { textView.text = "Échec de connexion" }
            }
        }
    }

    private inner class ConnectedThread(private val socket: BluetoothSocket) : Thread() {
        private val inputStream: InputStream = socket.inputStream
        private val buffer = ByteArray(1024)

        override fun run() {
            try {
                while (true) {
                    val bytes = inputStream.read(buffer)
                    val received = String(buffer, 0, bytes)
                    Log.i(TAG, "Reçu: $received")
                }
            } catch (e: IOException) {
                Log.d(TAG, "Connexion perdue", e)
            }
        }
    }

    private fun connecterAppareil(device: BluetoothDevice) {
        if (!verifierPermissions()) return
        if (!bluetoothAdapter.isEnabled) {
            Toast.makeText(activity, "Bluetooth non activé", Toast.LENGTH_SHORT).show()
            return
        }

        if (device.bondState != BluetoothDevice.BOND_BONDED) {
            device.createBond()
            Toast.makeText(activity, "Appairage en cours...", Toast.LENGTH_SHORT).show()
            return
        }

        Toast.makeText(activity, "Connexion...", Toast.LENGTH_SHORT).show()
        ConnectThread(device).start()
    }

    private fun deconnecterAppareil() {
        try {
            bluetoothSocket?.close()
            bluetoothSocket = null
            deviceConnected = null
            textView.text = "Déconnecté"
            Toast.makeText(activity, "Déconnecté", Toast.LENGTH_SHORT).show()
        } catch (e: IOException) {
            Toast.makeText(activity, "Erreur déconnexion", Toast.LENGTH_SHORT).show()
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
            requireContext().unregisterReceiver(bondReceiver)
        } catch (_: Exception) {}
    }
}
