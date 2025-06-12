package com.example.test

import android.Manifest
import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class Fragment1 : Fragment() {

    private lateinit var textViewBluetoothStatus: TextView
    private lateinit var btnPermissionsBT: Button
    private lateinit var btnActiverBT: Button
    private lateinit var btnRechercherAppareils: Button
    private lateinit var bluetoothAdapter: BluetoothAdapter
    private lateinit var recyclerViewDevices: RecyclerView
    private lateinit var deviceAdapter: BluetoothDeviceAdapter
    private val discoveredDevices = mutableListOf<BluetoothDevice>()

    private var listener: OnFragmentInteractionListener? = null

    private val requestPermissionLauncher: ActivityResultLauncher<Array<String>> = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.values.all { it }) {
            Toast.makeText(requireContext(), "Permissions OK", Toast.LENGTH_SHORT).show()
            updateBluetoothStatus()
        } else {
            Toast.makeText(requireContext(), "Permissions refusées", Toast.LENGTH_SHORT).show()
        }
    }

    private val requestEnableBtLauncher: ActivityResultLauncher<Intent> = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            Toast.makeText(requireContext(), "Bluetooth activé !", Toast.LENGTH_SHORT).show()
            updateBluetoothStatus()
        } else {
            Toast.makeText(requireContext(), "L\'utilisateur a refusé", Toast.LENGTH_SHORT).show()
        }
    }

    private val bluetoothReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                BluetoothDevice.ACTION_FOUND -> {
                    val device: BluetoothDevice? = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE)
                    device?.let { discoveredDevices.add(it) }
                    device?.let { deviceAdapter.addDevice(it) }
                    Log.d("Fragment1", "Device found: ${device?.name} - ${device?.address}")
                }
                BluetoothAdapter.ACTION_DISCOVERY_FINISHED -> {
                    Toast.makeText(requireContext(), "Recherche terminée", Toast.LENGTH_SHORT).show()
                    btnRechercherAppareils.isEnabled = true
                }
            }
        }
    }

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is OnFragmentInteractionListener) {
            listener = context
        } else {
            throw RuntimeException("$context must implement OnFragmentInteractionListener")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val filter = IntentFilter()
        filter.addAction(BluetoothDevice.ACTION_FOUND)
        filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED)
        requireActivity().registerReceiver(bluetoothReceiver, filter)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_1, container, false)

        textViewBluetoothStatus = view.findViewById(R.id.textViewBluetoothStatus)
        btnPermissionsBT = view.findViewById(R.id.buttonPermissionsBT)
        btnActiverBT = view.findViewById(R.id.buttonActiverBT)
        btnRechercherAppareils = view.findViewById(R.id.buttonRechercherAppareils)
        recyclerViewDevices = view.findViewById(R.id.recyclerViewDevices)

        recyclerViewDevices.layoutManager = LinearLayoutManager(context)
        deviceAdapter = BluetoothDeviceAdapter(discoveredDevices)
        recyclerViewDevices.adapter = deviceAdapter

        val bluetoothManager: BluetoothManager? = ContextCompat.getSystemService(requireContext(), BluetoothManager::class.java)
        bluetoothAdapter = bluetoothManager?.adapter ?: run {
            Toast.makeText(requireContext(), "L\'appareil ne supporte pas le Bluetooth", Toast.LENGTH_LONG).show()
            return null
        }

        btnPermissionsBT.setOnClickListener {
            requestBluetoothPermissions()
        }

        btnActiverBT.setOnClickListener {
            if (bluetoothAdapter.isEnabled) {
                Toast.makeText(requireContext(), "Déjà activé", Toast.LENGTH_SHORT).show()
            } else {
                val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
                requestEnableBtLauncher.launch(enableBtIntent)
            }
        }

        btnRechercherAppareils.setOnClickListener {
            startDiscovery()
        }

        view.findViewById<Button>(R.id.button_back).setOnClickListener { listener?.onFragmentInteraction(0) }
        view.findViewById<Button>(R.id.button_next).setOnClickListener { listener?.onFragmentInteraction(1) }

        updateBluetoothStatus()

        return view
    }

    override fun onDestroyView() {
        super.onDestroyView()
        requireActivity().unregisterReceiver(bluetoothReceiver)
        if (bluetoothAdapter.isDiscovering) {
            bluetoothAdapter.cancelDiscovery()
        }
    }

    override fun onDetach() {
        super.onDetach()
        listener = null
    }

    private fun requestBluetoothPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.BLUETOOTH_ADMIN,
            Manifest.permission.BLUETOOTH
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissions.add(Manifest.permission.BLUETOOTH_SCAN)
            permissions.add(Manifest.permission.BLUETOOTH_CONNECT)
            permissions.add(Manifest.permission.BLUETOOTH_ADVERTISE)
        }

        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.R) {
            permissions.add(Manifest.permission.ACCESS_FINE_LOCATION)
        }

        requestPermissionLauncher.launch(permissions.toTypedArray())
    }

    private fun updateBluetoothStatus() {
        if (bluetoothAdapter.isEnabled) {
            textViewBluetoothStatus.text = "Bluetooth activé et permissions OK"
        } else {
            textViewBluetoothStatus.text = "Bluetooth désactivé"
        }
    }

    private fun startDiscovery() {
        if (!bluetoothAdapter.isEnabled) {
            Toast.makeText(requireContext(), "Veuillez activer le Bluetooth d'abord", Toast.LENGTH_SHORT).show()
            return
        }
        if (bluetoothAdapter.isDiscovering) {
            bluetoothAdapter.cancelDiscovery()
        }
        deviceAdapter.clearDevices()
        discoveredDevices.clear()
        btnRechercherAppareils.isEnabled = false
        Toast.makeText(requireContext(), "Recherche des appareils...", Toast.LENGTH_SHORT).show()
        bluetoothAdapter.startDiscovery()
    }

    interface OnFragmentInteractionListener {
        fun onFragmentInteraction(direction: Int) // 0 for back, 1 for next
    }
} 